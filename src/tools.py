"""
src/tools.py
~~~~~~~~~~~~
Agent 工具定义与实现。

将现有的 PDF 转换、章节切分等功能封装为 agent 可调用的工具，
同时提供段落级读写工具，使 agent 能够逐段进行校对。

工具列表：
  - convert_pdf           : PDF → 原始 Markdown
  - split_chapters        : 原始 Markdown → 章节文件
  - load_existing_chapters: 加载已有章节目录（跳过转换/切分）
  - write_plan            : 写入 plan.md 校对计划
  - list_chapters         : 列出所有章节及校对状态
  - get_chapter_segment   : 获取某章的某个待校对段落
  - save_proofread_segment: 保存校对后的段落
  - finalize_chapter      : 合并段落并输出校对后的章节文件
  - complete              : 标记任务完成并输出摘要
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import cfg
from src.converter import convert_pdf
from src.splitter import Chapter, split_chapters, save_chapters
from src.proofreader import chunk_markdown

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI function-calling 工具描述
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "convert_pdf",
            "description": (
                "调用 OCR 模型将 PDF 文件转换为原始 Markdown 文本。"
                "转换后的 Markdown 会保存到 raw/full.md。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pdf_path": {
                        "type": "string",
                        "description": "PDF 文件路径",
                    },
                    "converter": {
                        "type": "string",
                        "enum": ["marker", "magic-pdf"],
                        "description": "转换后端，默认 marker",
                    },
                },
                "required": ["pdf_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "split_chapters",
            "description": (
                "将原始 Markdown（raw/full.md）按章节标题切分为独立章节文件，"
                "保存到 chapters/ 目录，并返回章节列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_md_path": {
                        "type": "string",
                        "description": "原始 Markdown 文件路径（如 raw/full.md）",
                    },
                },
                "required": ["raw_md_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_existing_chapters",
            "description": (
                "从已有章节目录加载 .md 文件，用于跳过转换/切分直接校对。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapters_dir": {
                        "type": "string",
                        "description": "包含 .md 章节文件的目录路径",
                    },
                },
                "required": ["chapters_dir"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_plan",
            "description": (
                "生成并写入 plan.md 校对计划文件。"
                "plan_content 应包含书名、章节列表、校对策略等信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_content": {
                        "type": "string",
                        "description": "plan.md 的完整 Markdown 内容",
                    },
                },
                "required": ["plan_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_chapters",
            "description": "列出所有章节及其校对进度（总段数、已校对段数、是否完成）。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chapter_segment",
            "description": (
                "获取某章节的指定段落原文，用于校对。"
                "返回段落文本及上下文（前一段末尾），便于连贯校对。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_file": {
                        "type": "string",
                        "description": "章节文件名（如 00_前言.md）",
                    },
                    "segment_index": {
                        "type": "integer",
                        "description": "段落序号（从 0 开始）",
                    },
                },
                "required": ["chapter_file", "segment_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_proofread_segment",
            "description": (
                "保存校对后的段落文本。"
                "corrected_text 必须是完整的校对后段落内容（不可省略或截断）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_file": {
                        "type": "string",
                        "description": "章节文件名",
                    },
                    "segment_index": {
                        "type": "integer",
                        "description": "段落序号",
                    },
                    "corrected_text": {
                        "type": "string",
                        "description": "校对后的完整段落文本",
                    },
                },
                "required": ["chapter_file", "segment_index", "corrected_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_chapter",
            "description": (
                "将该章所有校对后的段落合并，保存为 proofread/ 目录下的最终章节文件。"
                "应在该章所有段落均已校对后调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chapter_file": {
                        "type": "string",
                        "description": "章节文件名",
                    },
                },
                "required": ["chapter_file"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete",
            "description": "标记整个校对任务已完成，并附上处理摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "处理摘要（Markdown 格式）",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具上下文：跨调用维护状态
# ─────────────────────────────────────────────────────────────────────────────

class ToolContext:
    """在整个 agent 会话期间维护工具执行所需的状态。"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.chapters_dir = self.output_dir / "chapters"
        self.proofread_dir = self.output_dir / "proofread"
        self.images_dir = self.output_dir / "images"

        # 运行状态
        self.chapters: list[Chapter] = []
        self.chapter_segments: dict[str, list[str]] = {}       # filename → segments
        self.proofread_segments: dict[str, dict[int, str]] = {}  # filename → {seg_idx: text}
        self.completed: bool = False

        # 确保目录存在
        for d in (self.output_dir, self.raw_dir, self.chapters_dir, self.proofread_dir, self.images_dir):
            d.mkdir(parents=True, exist_ok=True)

    def ensure_segments(self, chapter_file: str) -> list[str]:
        """确保指定章节的段落已计算，返回段落列表。"""
        if chapter_file not in self.chapter_segments:
            path = self.chapters_dir / chapter_file
            if not path.exists():
                raise FileNotFoundError(f"章节文件不存在：{path}")
            content = path.read_text(encoding="utf-8")
            self.chapter_segments[chapter_file] = chunk_markdown(content, cfg.chunk_tokens)
            self.proofread_segments.setdefault(chapter_file, {})
        return self.chapter_segments[chapter_file]


# ─────────────────────────────────────────────────────────────────────────────
# 工具执行分发
# ─────────────────────────────────────────────────────────────────────────────

def execute_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> str:
    """按名称执行工具，返回 JSON 字符串结果。"""
    handlers = {
        "convert_pdf": _tool_convert_pdf,
        "split_chapters": _tool_split_chapters,
        "load_existing_chapters": _tool_load_existing_chapters,
        "write_plan": _tool_write_plan,
        "list_chapters": _tool_list_chapters,
        "get_chapter_segment": _tool_get_chapter_segment,
        "save_proofread_segment": _tool_save_proofread_segment,
        "finalize_chapter": _tool_finalize_chapter,
        "complete": _tool_complete,
    }

    handler = handlers.get(name)
    if handler is None:
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)

    try:
        result = handler(ctx, **arguments)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("工具 %s 执行失败：%s", name, e, exc_info=True)
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# 各工具实现
# ─────────────────────────────────────────────────────────────────────────────

def _tool_convert_pdf(ctx: ToolContext, pdf_path: str, converter: str = "marker") -> dict:
    raw_md, _img_dir, meta = convert_pdf(pdf_path, str(ctx.output_dir), converter)
    raw_file = ctx.raw_dir / "full.md"
    raw_file.write_text(raw_md, encoding="utf-8")
    logger.info("原始 Markdown 已保存：%s（%d 字符）", raw_file, len(raw_md))
    return {
        "status": "success",
        "raw_md_path": str(raw_file),
        "char_count": len(raw_md),
        "converter": meta.get("converter", converter),
    }


def _tool_split_chapters(ctx: ToolContext, raw_md_path: str) -> dict:
    raw_md = Path(raw_md_path).read_text(encoding="utf-8")
    chapters = split_chapters(raw_md)
    save_chapters(chapters, str(ctx.chapters_dir))
    ctx.chapters = chapters
    logger.info("切分完成：共 %d 章", len(chapters))

    chapter_list = []
    for ch in chapters:
        segments = chunk_markdown(ch.content, cfg.chunk_tokens)
        ctx.chapter_segments[ch.filename] = segments
        ctx.proofread_segments[ch.filename] = {}
        chapter_list.append({
            "index": ch.index,
            "title": ch.title,
            "filename": ch.filename,
            "char_count": len(ch.content),
            "segment_count": len(segments),
        })

    return {
        "status": "success",
        "total_chapters": len(chapters),
        "chapters": chapter_list,
    }


def _tool_load_existing_chapters(ctx: ToolContext, chapters_dir: str) -> dict:
    src_dir = Path(chapters_dir)
    if not src_dir.exists():
        return {"error": f"目录不存在：{chapters_dir}"}

    md_files = sorted(src_dir.glob("*.md"))
    if not md_files:
        return {"error": f"目录中未找到 .md 文件：{chapters_dir}"}

    chapters: list[Chapter] = []
    chapter_list: list[dict] = []

    for i, p in enumerate(md_files):
        content = p.read_text(encoding="utf-8")
        ch = Chapter(index=i, title=p.stem, content=content, slug=p.stem)
        chapters.append(ch)

        # 同时写入 ctx.chapters_dir（以便后续工具读取）
        dest = ctx.chapters_dir / ch.filename
        if not dest.exists():
            dest.write_text(content, encoding="utf-8")

        segments = chunk_markdown(content, cfg.chunk_tokens)
        ctx.chapter_segments[ch.filename] = segments
        ctx.proofread_segments[ch.filename] = {}
        chapter_list.append({
            "index": ch.index,
            "title": ch.title,
            "filename": ch.filename,
            "char_count": len(content),
            "segment_count": len(segments),
        })

    ctx.chapters = chapters
    logger.info("从 %s 加载了 %d 个章节", chapters_dir, len(chapters))
    return {
        "status": "success",
        "total_chapters": len(chapters),
        "chapters": chapter_list,
    }


def _tool_write_plan(ctx: ToolContext, plan_content: str) -> dict:
    plan_path = ctx.output_dir / "plan.md"
    plan_path.write_text(plan_content, encoding="utf-8")
    logger.info("校对计划已写入：%s", plan_path)
    return {"status": "success", "plan_path": str(plan_path)}


def _tool_list_chapters(ctx: ToolContext) -> dict:
    chapters_info = []
    for ch in ctx.chapters:
        segments = ctx.chapter_segments.get(ch.filename, [])
        proofread = ctx.proofread_segments.get(ch.filename, {})
        chapters_info.append({
            "index": ch.index,
            "title": ch.title,
            "filename": ch.filename,
            "total_segments": len(segments),
            "proofread_segments": len(proofread),
            "completed": len(proofread) == len(segments) and len(segments) > 0,
        })
    return {"chapters": chapters_info}


def _tool_get_chapter_segment(ctx: ToolContext, chapter_file: str, segment_index: int) -> dict:
    segments = ctx.ensure_segments(chapter_file)
    if segment_index < 0 or segment_index >= len(segments):
        return {"error": f"段落序号 {segment_index} 超出范围 [0, {len(segments)})"}

    # 提供前一段的末尾作为上下文
    proofread = ctx.proofread_segments.get(chapter_file, {})
    context = ""
    if segment_index > 0:
        prev_text = proofread.get(segment_index - 1, segments[segment_index - 1])
        context = prev_text[-400:] if len(prev_text) > 400 else prev_text

    return {
        "chapter_file": chapter_file,
        "segment_index": segment_index,
        "total_segments": len(segments),
        "context": context,
        "segment_text": segments[segment_index],
    }


def _tool_save_proofread_segment(
    ctx: ToolContext,
    chapter_file: str,
    segment_index: int,
    corrected_text: str,
) -> dict:
    segments = ctx.ensure_segments(chapter_file)
    if segment_index < 0 or segment_index >= len(segments):
        return {"error": f"段落序号 {segment_index} 超出范围 [0, {len(segments)})"}

    ctx.proofread_segments.setdefault(chapter_file, {})[segment_index] = corrected_text

    proofread = ctx.proofread_segments[chapter_file]
    return {
        "status": "success",
        "chapter_file": chapter_file,
        "segment_index": segment_index,
        "proofread_count": len(proofread),
        "total_segments": len(segments),
        "chapter_complete": len(proofread) == len(segments),
    }


def _tool_finalize_chapter(ctx: ToolContext, chapter_file: str) -> dict:
    segments = ctx.ensure_segments(chapter_file)
    proofread = ctx.proofread_segments.get(chapter_file, {})

    # 合并段落：优先使用校对版本，否则保留原文
    parts = []
    for i in range(len(segments)):
        parts.append(proofread.get(i, segments[i]))

    final_text = "\n\n".join(parts)
    dest = ctx.proofread_dir / chapter_file
    dest.write_text(final_text, encoding="utf-8")

    unclear_cnt = final_text.count(cfg.unclear_marker)
    logger.info(
        "章节 %s 校对完成：%d 字符，%d 处 <unclear/>",
        chapter_file, len(final_text), unclear_cnt,
    )
    return {
        "status": "success",
        "chapter_file": chapter_file,
        "output_path": str(dest),
        "char_count": len(final_text),
        "unclear_count": unclear_cnt,
        "segments_proofread": len(proofread),
        "total_segments": len(segments),
    }


def _tool_complete(ctx: ToolContext, summary: str = "") -> dict:
    ctx.completed = True

    # 写入摘要
    summary_path = ctx.output_dir / "summary.md"
    if summary:
        summary_path.write_text(summary, encoding="utf-8")
        logger.info("处理摘要已写入：%s", summary_path)

    return {
        "status": "complete",
        "output_dir": str(ctx.output_dir),
        "summary_path": str(summary_path) if summary else None,
    }
