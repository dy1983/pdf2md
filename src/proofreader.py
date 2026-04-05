"""
src/proofreader.py
~~~~~~~~~~~~~~~~~~
调用 LLM 对每个章节 Markdown 进行校对：
  - 修正 OCR 识别错误（字符混淆、上下文语义错误等）
  - 修复断行（OCR 将句子切断成多行）
  - 恢复段落结构
  - 保留 LaTeX 公式、Markdown 语法、图片引用、代码块
  - 对极度模糊/逻辑不通的段落插入 <unclear/> 标记

输出：
  ProofResult
    corrected   : str   – 校对后的 Markdown
    confidence  : int   – 0~100，对本章整体质量的置信度评估
    changes     : list  – LLM 报告的主要修改列表
    unclear_cnt : int   – 插入 <unclear/> 的次数
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一位专业学术书稿校对编辑，专门修正从 PDF OCR 提取的 Markdown 文本错误。

【你的任务】
1. 修正 OCR 识别错误，包括但不限于：
   - 字母/数字混淆：如"1"↔"l"/"I"，"0"↔"O"，"rn"↔"m"，"cl"↔"d"
   - 中文形近字/同音字误识别（需结合语义判断）
   - 标点符号错误（如全半角混用、引号不匹配）
2. 修复 OCR 断行错误：将被错误分割成多行的句子重新拼合为完整段落。
3. 恢复正确的段落与章节结构。
4. 保留书中原有的专业术语、人名、地名，不擅自改写。

【必须原样保留，绝对不得修改】
- 所有 LaTeX 公式：行内 $...$、行间 $$...$$、\\[...\\]、\\(...\\)
- 所有化学式（如 $\\mathrm{H_2O}$）
- 所有 Markdown 语法：标题（#）、**粗体**、*斜体*、列表（-/*）、表格（|）、代码块（```）、链接、图片 ![...](...) 
- 所有代码块内容

【不确定性标记】
若某段文字极度模糊、无法读通，或逻辑完全断裂（非 OCR 错误可修复），
在该段落末尾（段落最后一行之后）插入一行：<unclear/>

【返回格式】
必须返回合法 JSON，格式如下（不含任何额外文字）：
{
  "corrected": "校对后的完整 Markdown 文本",
  "confidence": 整数（0-100，表示对本片段校对质量的置信度）,
  "changes": ["改动描述1", "改动描述2", ...]
}
"""

USER_TEMPLATE = """\
【上下文（仅供参考，无需校对）】
{context}

【待校对片段】
{chunk}
"""


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkResult:
    corrected: str
    confidence: int
    changes: list[str] = field(default_factory=list)


@dataclass
class ProofResult:
    corrected: str
    confidence: int            # 章节平均置信度
    changes: list[str]         # 所有 chunk 的修改摘要
    unclear_cnt: int           # <unclear/> 出现次数
    chunk_count: int           # 处理的 chunk 数量
    elapsed_sec: float         # 耗时（秒）


# ─────────────────────────────────────────────────────────────────────────────
# Token 计数（无 tiktoken 时降级到字符估算）
# ─────────────────────────────────────────────────────────────────────────────

def _count_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # 中文约 1.5 字/token，英文约 4 字符/token，取保守估计 2 字符/token
        return len(text) // 2


# ─────────────────────────────────────────────────────────────────────────────
# 保护性分块：避免在 LaTeX / 代码块 / 表格中间切断
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_paragraphs(text: str) -> list[str]:
    """按空行切分段落，保留空行作为分隔符。"""
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n{2,}", text)
    return [p for p in parts if p.strip()]


def _is_in_protected_block(para: str) -> bool:
    """简单检测该段落是否为"整块"保护内容（代码块 / 数学块 / 表格行）。"""
    stripped = para.strip()
    # 代码块
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return True
    # 独立数学块
    if stripped.startswith("$$") or stripped.startswith("\\["):
        return True
    # Markdown 表格行
    if stripped.startswith("|"):
        return True
    return False


def chunk_markdown(text: str, max_tokens: int) -> list[str]:
    """
    将 Markdown 切分为不超过 max_tokens 的块，
    尽量在段落边界切分，保护代码块 / 数学块 / 表格。
    """
    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    # 处理"受保护"的多行块（代码 / $$...$$）
    merged: list[str] = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        stripped = para.strip()
        # 多行代码块：寻找配对的闭合标记
        if stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            block_lines = [para]
            i += 1
            while i < len(paragraphs):
                block_lines.append(paragraphs[i])
                if paragraphs[i].strip().startswith(fence) and len(paragraphs[i].strip()) == 3:
                    i += 1
                    break
                i += 1
            merged.append("\n\n".join(block_lines))
            continue
        # 行间数学块
        if stripped.startswith("$$") and not stripped.endswith("$$"):
            block_lines = [para]
            i += 1
            while i < len(paragraphs):
                block_lines.append(paragraphs[i])
                if paragraphs[i].strip().endswith("$$"):
                    i += 1
                    break
                i += 1
            merged.append("\n\n".join(block_lines))
            continue
        merged.append(para)
        i += 1

    # 按 token 数量打包
    for para in merged:
        tok = _count_tokens(para)
        if current_tokens + tok > max_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_tokens = tok
        else:
            current_parts.append(para)
            current_tokens += tok

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────────────────────────────────────

def _call_llm(
    client: OpenAI,
    chunk: str,
    context: str,
    model: str,
    max_retries: int,
) -> ChunkResult:
    """调用 LLM 校对单个 chunk，失败时重试，最终降级返回原文。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                context=context[-300:] if context else "（无上文）",
                chunk=chunk,
            ),
        },
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            return ChunkResult(
                corrected=data.get("corrected", chunk),
                confidence=int(data.get("confidence", 80)),
                changes=data.get("changes", []),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM 返回 JSON 解析失败（尝试 %d/%d）：%s", attempt, max_retries, e)
        except RateLimitError:
            wait = 2 ** attempt
            logger.warning("触发速率限制，等待 %ds 后重试…", wait)
            time.sleep(wait)
        except APIError as e:
            logger.warning("LLM API 错误（尝试 %d/%d）：%s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(1)

    # 全部重试失败，保留原文
    logger.error("chunk 校对失败，保留原文")
    return ChunkResult(corrected=chunk, confidence=50, changes=["[校对失败，保留原文]"])


# ─────────────────────────────────────────────────────────────────────────────
# 公开接口
# ─────────────────────────────────────────────────────────────────────────────

def proofread_chapter(
    chapter_md: str,
    chapter_title: str = "",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    chunk_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> ProofResult:
    """
    对单个章节 Markdown 进行 LLM 校对。

    参数
    ----
    chapter_md    : 待校对的 Markdown 文本
    chapter_title : 章节标题（仅用于日志）
    其余参数若为 None，则从 config.cfg 读取
    """
    _api_key     = api_key     or cfg.llm_api_key
    _base_url    = base_url    or cfg.llm_base_url
    _model       = model       or cfg.llm_model
    _chunk_tok   = chunk_tokens or cfg.chunk_tokens
    _retries     = max_retries if max_retries is not None else cfg.llm_max_retries

    if not _api_key:
        raise ValueError("未设置 OPENAI_API_KEY，请在 .env 中配置")

    client = OpenAI(api_key=_api_key, base_url=_base_url)

    chunks = chunk_markdown(chapter_md, _chunk_tok)
    logger.info("章节「%s」共 %d 个 chunk，开始校对…", chapter_title, len(chunks))

    corrected_parts: list[str] = []
    all_changes: list[str] = []
    confidences: list[int] = []
    context_so_far = ""
    t0 = time.time()

    for idx, chunk in enumerate(chunks, 1):
        logger.debug("  校对 chunk %d/%d（%d token）…", idx, len(chunks), _count_tokens(chunk))
        result = _call_llm(client, chunk, context_so_far, _model, _retries)
        corrected_parts.append(result.corrected)
        all_changes.extend(result.changes)
        confidences.append(result.confidence)
        # 将本 chunk 末尾作为下一 chunk 的上下文
        context_so_far = result.corrected[-400:]

    final_text = "\n\n".join(corrected_parts)
    unclear_cnt = final_text.count(cfg.unclear_marker)
    avg_conf = round(sum(confidences) / len(confidences)) if confidences else 0

    return ProofResult(
        corrected=final_text,
        confidence=avg_conf,
        changes=all_changes,
        unclear_cnt=unclear_cnt,
        chunk_count=len(chunks),
        elapsed_sec=time.time() - t0,
    )
