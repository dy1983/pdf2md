"""
src/splitter.py
~~~~~~~~~~~~~~~
将全书 Markdown 按章节标题切分为多个独立章节对象。

识别策略：
  1. 遍历每一行，用配置中的正则列表匹配一级/二级标题。
  2. 匹配到新标题时，将之前累积的内容归为一个 Chapter。
  3. 若未找到任何标题，整篇文档作为单一章节返回。

Chapter 命名规则：
  序号两位补零 + "_" + 首行标题（去除 # 号，最多 40 字符，非法文件名字符替换为 _）
  例：00_前言.md、01_第一章_绪论.md
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config import cfg


@dataclass
class Chapter:
    index: int          # 序号，从 0 开始
    title: str          # 标题原文（去掉开头 # 空格后的内容）
    content: str        # Markdown 正文（包含标题行）
    slug: str = ""      # 用于文件名的安全字符串

    def __post_init__(self) -> None:
        if not self.slug:
            self.slug = _make_slug(self.index, self.title)

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────────────────────────────────────

def _make_slug(index: int, title: str) -> str:
    """生成文件名友好的 slug，最多 50 个字符。"""
    # 去除 Markdown # 前缀与空白
    clean = re.sub(r"^#+\s*", "", title).strip()
    # 截断
    clean = clean[:40]
    # 将非法文件名字符（Windows + macOS 限制）替换为 _
    clean = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", clean)
    # 将连续空白替换为单 _
    clean = re.sub(r"\s+", "_", clean).strip("_")
    return f"{index:02d}_{clean}" if clean else f"{index:02d}_chapter"


def _is_chapter_heading(line: str, patterns: Sequence[str]) -> bool:
    """判断某行是否为章节标题。"""
    stripped = line.strip()
    if not stripped:
        return False
    for pat in patterns:
        if re.match(pat, stripped, re.IGNORECASE):
            return True
    return False


def _extract_title(line: str) -> str:
    """从标题行提取纯文本（去除 # 前缀）。"""
    return re.sub(r"^#+\s*", "", line).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────────────────

def split_chapters(
    markdown: str,
    patterns: Sequence[str] | None = None,
) -> list[Chapter]:
    """
    将完整 Markdown 文本切分为章节列表。

    参数
    ----
    markdown : 全文 Markdown 字符串
    patterns : 章节标题正则列表（None 时使用 config 中的默认值）

    返回
    ----
    list[Chapter]  按顺序排列，至少包含一个元素
    """
    if patterns is None:
        patterns = cfg.chapter_patterns

    lines = markdown.splitlines(keepends=True)
    chapters: list[Chapter] = []

    current_title: str = "前言"
    current_lines: list[str] = []
    index = 0

    for line in lines:
        if _is_chapter_heading(line, patterns):
            # 保存上一章（如果有内容）
            content = "".join(current_lines).strip()
            if content:
                chapters.append(Chapter(index=index, title=current_title, content=content))
                index += 1
            # 开始新章
            current_title = _extract_title(line)
            current_lines = [line]
        else:
            current_lines.append(line)

    # 最后一章
    content = "".join(current_lines).strip()
    if content:
        chapters.append(Chapter(index=index, title=current_title, content=content))

    # 若完全没有匹配到标题，作为单章返回
    if not chapters:
        chapters.append(
            Chapter(index=0, title="全文", content=markdown.strip())
        )

    return chapters


def save_chapters(chapters: list[Chapter], output_dir: str | Path) -> list[Path]:
    """
    将章节列表写入磁盘，返回写入的文件路径列表。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for ch in chapters:
        p = out / ch.filename
        p.write_text(ch.content, encoding="utf-8")
        paths.append(p)

    return paths
