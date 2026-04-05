"""
src/pipeline.py
~~~~~~~~~~~~~~~
主流程编排：PDF → Markdown → 章节切分 → LLM 校对 → 输出

输出目录结构：
  <book_name>/
    images/              ← 原始图片（由 converter 填充）
    raw/                 ← 转换器直出的原始 Markdown（原始备份）
      full.md
    chapters/            ← 切分后、校对前的章节文件
      00_前言.md
      01_第一章_xxx.md
      ...
    proofread/           ← LLM 校对后的最终章节文件（主要输出）
      00_前言.md
      01_第一章_xxx.md
      ...
    summary.txt          ← 处理摘要
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import cfg
from src.converter import convert_pdf
from src.splitter import Chapter, split_chapters, save_chapters
from src.proofreader import ProofResult, proofread_chapter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChapterSummary:
    index: int
    title: str
    filename: str
    char_count: int
    chunk_count: int
    confidence: int
    unclear_cnt: int
    elapsed_sec: float
    skipped: bool = False          # 跳过校对时为 True


@dataclass
class PipelineResult:
    book_name: str
    output_dir: Path
    converter: str
    total_chapters: int
    chapter_summaries: list[ChapterSummary] = field(default_factory=list)
    convert_elapsed: float = 0.0
    total_elapsed: float = 0.0
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# 摘要写入
# ─────────────────────────────────────────────────────────────────────────────

def _write_summary(result: PipelineResult, summary_path: Path) -> None:
    lines: list[str] = [
        "=" * 60,
        f"  PDF → Markdown 转换摘要",
        "=" * 60,
        f"书名         : {result.book_name}",
        f"输出目录     : {result.output_dir}",
        f"转换器       : {result.converter}",
        f"处理时间     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"转换耗时     : {result.convert_elapsed:.1f}s",
        f"总耗时       : {result.total_elapsed:.1f}s",
        f"章节总数     : {result.total_chapters}",
        "",
        "-" * 60,
        "  章节详情",
        "-" * 60,
    ]

    for s in result.chapter_summaries:
        status = "（跳过校对）" if s.skipped else ""
        lines += [
            f"",
            f"  [{s.index:02d}] {s.title} {status}",
            f"       文件名  : {s.filename}",
            f"       字符数  : {s.char_count:,}",
            f"       chunks  : {s.chunk_count}",
            f"       置信度  : {s.confidence}%",
            f"       <unclear/> : {s.unclear_cnt} 处",
            f"       耗时    : {s.elapsed_sec:.1f}s",
        ]

    if result.chapter_summaries:
        avg_conf = round(
            sum(s.confidence for s in result.chapter_summaries) / len(result.chapter_summaries)
        )
        lines += [
            "",
            "-" * 60,
            f"  全书平均置信度 : {avg_conf}%",
            f"  <unclear/> 总计: {sum(s.unclear_cnt for s in result.chapter_summaries)} 处",
            "-" * 60,
        ]

    if result.error:
        lines += ["", f"[错误] {result.error}"]

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("摘要已写入：%s", summary_path)


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    output_root: str = "output",
    book_name: Optional[str] = None,
    *,
    converter: Optional[str] = None,
    skip_proofread: bool = False,
    proofread_only_dir: Optional[str] = None,
    on_chapter_start=None,    # callback(index, total, title)
    on_chapter_done=None,     # callback(ChapterSummary)
) -> PipelineResult:
    """
    端到端执行 PDF 转换管线。

    参数
    ----
    pdf_path         : 输入 PDF 路径
    output_root      : 所有书籍输出的根目录（默认 "output"）
    book_name        : 书名（用于输出子目录命名，默认取 PDF 文件名）
    converter        : "marker" | "magic-pdf"（覆盖 config）
    skip_proofread   : 跳过 LLM 校对步骤
    proofread_only_dir : 仅校对已有章节目录（跳过 PDF 转换与切分）
    on_chapter_start : 进度回调
    on_chapter_done  : 进度回调
    """
    t_total = time.time()
    _converter = converter or cfg.converter

    # ── 确定书名与输出目录 ────────────────────────────────────────
    if book_name is None:
        book_name = Path(pdf_path).stem
    out_dir = Path(output_root) / book_name
    images_dir = out_dir / "images"
    raw_dir = out_dir / "raw"
    chapters_dir = out_dir / "chapters"
    proofread_dir = out_dir / "proofread"

    for d in (out_dir, raw_dir, chapters_dir, proofread_dir):
        d.mkdir(parents=True, exist_ok=True)

    result = PipelineResult(
        book_name=book_name,
        output_dir=out_dir,
        converter=_converter,
        total_chapters=0,
    )

    # ─────────────────────────────────────────────────────────────
    # 步骤 1：PDF → Markdown（可跳过，直接从现有章节目录校对）
    # ─────────────────────────────────────────────────────────────
    if proofread_only_dir:
        logger.info("跳过转换，直接从目录 %s 读取章节…", proofread_only_dir)
        chapters_dir = Path(proofread_only_dir)
        chapters = [
            Chapter(
                index=i,
                title=p.stem,
                content=p.read_text(encoding="utf-8"),
            )
            for i, p in enumerate(sorted(chapters_dir.glob("*.md")))
        ]
        result.convert_elapsed = 0.0
    else:
        logger.info("开始转换 PDF：%s", pdf_path)
        t_conv = time.time()
        try:
            raw_md, _img_dir, meta = convert_pdf(pdf_path, str(out_dir), _converter)
        except Exception as e:
            result.error = f"PDF 转换失败：{e}"
            logger.error(result.error)
            _write_summary(result, out_dir / "summary.txt")
            return result
        result.convert_elapsed = time.time() - t_conv
        result.converter = meta.get("converter", _converter)

        # 保存原始 Markdown
        raw_file = raw_dir / "full.md"
        raw_file.write_text(raw_md, encoding="utf-8")
        logger.info("原始 Markdown 已保存：%s (%d 字符)", raw_file, len(raw_md))

        # 切分章节
        chapters = split_chapters(raw_md)
        save_chapters(chapters, str(chapters_dir))
        logger.info("切分完成：共 %d 章，已保存至 %s", len(chapters), chapters_dir)

    result.total_chapters = len(chapters)

    # ─────────────────────────────────────────────────────────────
    # 步骤 2：LLM 校对
    # ─────────────────────────────────────────────────────────────
    for ch in chapters:
        if on_chapter_start:
            on_chapter_start(ch.index, len(chapters), ch.title)

        dest = proofread_dir / ch.filename

        if skip_proofread:
            # 直接复制原章节
            shutil.copy2(chapters_dir / ch.filename, dest)
            summary = ChapterSummary(
                index=ch.index,
                title=ch.title,
                filename=ch.filename,
                char_count=len(ch.content),
                chunk_count=0,
                confidence=100,
                unclear_cnt=0,
                elapsed_sec=0.0,
                skipped=True,
            )
        else:
            logger.info("校对章节 [%02d] %s …", ch.index, ch.title)
            try:
                proof: ProofResult = proofread_chapter(ch.content, ch.title)
                dest.write_text(proof.corrected, encoding="utf-8")
                summary = ChapterSummary(
                    index=ch.index,
                    title=ch.title,
                    filename=ch.filename,
                    char_count=len(proof.corrected),
                    chunk_count=proof.chunk_count,
                    confidence=proof.confidence,
                    unclear_cnt=proof.unclear_cnt,
                    elapsed_sec=proof.elapsed_sec,
                )
                logger.info(
                    "  ✓ 完成，置信度 %d%%，<unclear/> %d 处，耗时 %.1fs",
                    proof.confidence,
                    proof.unclear_cnt,
                    proof.elapsed_sec,
                )
            except Exception as e:
                logger.error("章节 [%02d] 校对失败：%s，保留原文", ch.index, e)
                shutil.copy2(chapters_dir / ch.filename, dest)
                summary = ChapterSummary(
                    index=ch.index,
                    title=ch.title,
                    filename=ch.filename,
                    char_count=len(ch.content),
                    chunk_count=0,
                    confidence=0,
                    unclear_cnt=0,
                    elapsed_sec=0.0,
                    skipped=True,
                )

        result.chapter_summaries.append(summary)
        if on_chapter_done:
            on_chapter_done(summary)

    result.total_elapsed = time.time() - t_total
    _write_summary(result, out_dir / "summary.txt")
    logger.info("管线完成，总耗时 %.1fs，输出目录：%s", result.total_elapsed, out_dir)
    return result
