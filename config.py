"""
全局配置：从 .env 加载，提供带类型的访问入口。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录的 .env
load_dotenv(Path(__file__).parent / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# 章节标题识别正则（按优先级排列）
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CHAPTER_PATTERNS: list[str] = [
    # Markdown 一级/二级标题 + 中文章节
    r"^#{1,2}\s*第\s*[〇零一二三四五六七八九十百千\d]+\s*章",
    # 无 # 号的中文章节独立行（"章"后必须紧跟空白或行尾，避免匹配"第一章的内容…"）
    r"^第\s*[〇零一二三四五六七八九十百千\d]+\s*章(?:\s|$)",
    # 英文 Chapter
    r"^#{1,2}\s*Chapter\s+\d+",
    # 纯数字 Markdown 标题（如"# 1 Introduction"）
    r"^#{1,2}\s*\d+[\.\s]+\S",
    # 前言 / 序言 / 附录 等（要求 # 前缀，避免正文内容误匹配）
    r"^#{1,2}\s*(前言|序言|绪论|引言|后记|附录|参考文献|Bibliography|Appendix|Preface|Introduction)",
]


@dataclass
class Config:
    # ── LLM ──────────────────────────────────────────────────────
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o"))

    # ── 转换器 ────────────────────────────────────────────────────
    converter: str = field(default_factory=lambda: os.getenv("CONVERTER", "marker"))

    # ── 校对参数 ──────────────────────────────────────────────────
    chunk_tokens: int = field(
        default_factory=lambda: int(os.getenv("PROOFREAD_CHUNK_TOKENS", "1200"))
    )
    concurrency: int = field(
        default_factory=lambda: int(os.getenv("PROOFREAD_CONCURRENCY", "3"))
    )
    llm_max_retries: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "3"))
    )

    # ── 章节识别 ──────────────────────────────────────────────────
    chapter_patterns: list[str] = field(default_factory=lambda: DEFAULT_CHAPTER_PATTERNS)

    # ── 输出 ──────────────────────────────────────────────────────
    # 不确定性标记（插入 Markdown 中）
    unclear_marker: str = "<unclear/>"


# 单例，供各模块直接 import
cfg = Config()
