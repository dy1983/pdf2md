"""
src/converter.py
~~~~~~~~~~~~~~~~
将 PDF 转换为粗略 Markdown，同时抽取内嵌图片。

支持两种后端：
  - marker-pdf  (默认，自动下载布局/OCR 模型)
  - magic-pdf   (MinerU，通过 CLI 调用)

返回值：
    markdown_text  : str          – Markdown 正文
    images_dir     : Path | None  – 图片保存目录（相对于 output_dir）
    meta           : dict         – 转换器返回的元数据
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

def _save_images(images: dict, images_dir: Path) -> dict[str, str]:
    """将 marker 返回的图片字典写到磁盘，返回 {原名: 相对路径}。"""
    images_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, img in images.items():
        dest = images_dir / name
        if hasattr(img, "save"):          # PIL.Image
            img.save(str(dest))
        elif isinstance(img, (bytes, bytearray)):
            dest.write_bytes(img)
        else:
            logger.warning("未知图片类型 %s，跳过", name)
            continue
        saved[name] = str(dest)
    return saved


def _fix_image_paths(md: str, images_dir: Path, output_dir: Path) -> str:
    """将 Markdown 中的图片路径统一替换为相对于 output_dir 的路径。"""
    rel = images_dir.relative_to(output_dir)

    def _replace(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        # 仅替换本地路径（非 http/https）
        if src.startswith(("http://", "https://")):
            return m.group(0)
        fname = Path(src).name
        return f"![{alt}]({rel}/{fname})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _replace, md)


# ─────────────────────────────────────────────────────────────────────────────
# marker-pdf 后端
# ─────────────────────────────────────────────────────────────────────────────

def _convert_marker(pdf_path: str, output_dir: Path) -> tuple[str, Path, dict]:
    images_dir = output_dir / "images"

    # marker v2 API
    try:
        from marker.converters.pdf import PdfConverter          # type: ignore
        from marker.models import create_model_dict              # type: ignore
        from marker.output import text_from_rendered             # type: ignore

        logger.info("使用 marker v2 API 加载模型…")
        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(pdf_path)
        md, _, images = text_from_rendered(rendered)
        meta = {"converter": "marker-v2"}
        _save_images(images, images_dir)
        md = _fix_image_paths(md, images_dir, output_dir)
        return md, images_dir, meta

    except (ImportError, AttributeError):
        pass

    # marker v1 API（旧版兼容）
    try:
        from marker.convert import convert_single_pdf            # type: ignore
        from marker.models import load_all_models                # type: ignore

        logger.info("使用 marker v1 API 加载模型…")
        models = load_all_models()
        md, images, meta_raw = convert_single_pdf(pdf_path, models)
        meta = dict(meta_raw) if meta_raw else {}
        meta["converter"] = "marker-v1"
        _save_images(images, images_dir)
        md = _fix_image_paths(md, images_dir, output_dir)
        return md, images_dir, meta

    except ImportError as exc:
        raise ImportError(
            "未找到 marker-pdf 库，请执行：pip install marker-pdf"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# magic-pdf (MinerU) 后端
# ─────────────────────────────────────────────────────────────────────────────

def _convert_magic_pdf(pdf_path: str, output_dir: Path) -> tuple[str, Path, dict]:
    if shutil.which("magic-pdf") is None:
        raise FileNotFoundError(
            "magic-pdf 命令未找到，请执行：pip install magic-pdf[full]"
        )

    logger.info("调用 magic-pdf CLI 转换 %s …", pdf_path)
    result = subprocess.run(
        ["magic-pdf", "-p", pdf_path, "-o", str(output_dir), "--method", "auto"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"magic-pdf 转换失败:\n{result.stderr}")

    # magic-pdf 在 output_dir/<pdf_stem>/auto/ 下生成 *.md
    pdf_stem = Path(pdf_path).stem
    md_files = sorted((output_dir / pdf_stem).rglob("*.md"))
    if not md_files:
        raise FileNotFoundError("magic-pdf 未生成 Markdown 文件，请检查日志")

    # 合并多个 md（通常只有一个）
    md = "\n\n".join(f.read_text(encoding="utf-8") for f in md_files)

    images_dir = output_dir / "images"
    # magic-pdf 的图片通常在 output_dir/<pdf_stem>/auto/images/
    src_images = output_dir / pdf_stem / "auto" / "images"
    if src_images.exists():
        shutil.copytree(src_images, images_dir, dirs_exist_ok=True)
        md = _fix_image_paths(md, images_dir, output_dir)

    return md, images_dir, {"converter": "magic-pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: str,
    output_dir: str,
    converter: str = "marker",
) -> tuple[str, Optional[Path], dict]:
    """
    参数
    ----
    pdf_path   : PDF 文件路径
    output_dir : 输出根目录（会自动创建）
    converter  : "marker" | "magic-pdf"

    返回
    ----
    (markdown_text, images_dir_or_None, metadata_dict)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    backends = (
        [_convert_marker, _convert_magic_pdf]
        if converter == "marker"
        else [_convert_magic_pdf, _convert_marker]
    )

    last_err: Exception = RuntimeError("无可用转换后端")
    for backend in backends:
        try:
            md, images_dir, meta = backend(pdf_path, out)
            logger.info("转换完成（后端：%s），Markdown 长度：%d 字符", meta.get("converter"), len(md))
            return md, images_dir, meta
        except (ImportError, FileNotFoundError, RuntimeError) as e:
            logger.warning("%s 不可用：%s，尝试备用后端…", backend.__name__, e)
            last_err = e

    raise last_err
