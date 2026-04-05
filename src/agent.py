"""
src/agent.py
~~~~~~~~~~~~
面向 Agent 的 PDF → Markdown 全流程编排。

Agent 使用 LLM 的 function calling 能力调用工具（PDF 转换、章节切分等），
并利用自身的语言理解能力直接完成校对，无需额外设置 LLM。

工作流程：
  1. 调用 convert_pdf 工具转换 PDF → 原始 Markdown
  2. 调用 split_chapters 工具切分章节
  3. 生成 plan.md，列举所有章节与校对策略
  4. 逐章、逐段读取并校对，分段处理避免上下文溢出
  5. 每章校对完成后合并保存，继续下一章，中途不停
  6. 全部完成后输出摘要
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional, Callable

from openai import OpenAI

from config import cfg
from src.tools import TOOL_SCHEMAS, ToolContext, execute_tool

logger = logging.getLogger(__name__)

# 每章校对完成后重置对话历史，避免上下文溢出
# 在一章内部，最多保留的消息数（超出时裁剪）
_MAX_MESSAGES_PER_CHAPTER = 60


# ─────────────────────────────────────────────────────────────────────────────
# Agent 系统提示词
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
你是 pdf2md Agent——一个专业的 PDF 书籍转 Markdown 助手。
你可以调用工具完成 PDF 转换和章节切分，并用你自己的语言能力直接完成校对。

## 工作流程

请严格按以下步骤执行（不要跳步）：

### 第一步：PDF 转换
调用 `convert_pdf` 工具，将 PDF 转为原始 Markdown。

### 第二步：章节切分
调用 `split_chapters` 工具，将原始 Markdown 切分为独立章节文件。

### 第三步：生成校对计划
调用 `write_plan` 工具生成 plan.md，内容包括：
- 书名
- 章节编号、标题、字符数、分段数列表（表格形式）
- 校对策略说明

### 第四步：逐章逐段校对
对每一章，按顺序执行：
1. 调用 `get_chapter_segment` 获取一个段落及其上下文
2. 仔细阅读段落内容，按照【校对规则】进行修正
3. 调用 `save_proofread_segment`，将完整的校对后段落作为 corrected_text 传入
4. 重复以上 1-3 步，直到该章所有段落校对完毕
5. 调用 `finalize_chapter` 合并并保存该章的最终文件
6. 继续下一章，中途不停

### 第五步：完成
所有章节校对完毕后，调用 `complete` 工具并附上处理摘要。

## 校对规则

### 必须修正的内容
1. **OCR 识别错误**：
   - 字母/数字混淆："1"↔"l"/"I"、"0"↔"O"、"rn"↔"m"、"cl"↔"d"
   - 中文形近字/同音字误识别（结合上下文语义判断）
   - 标点符号错误（全半角混用、引号不匹配）
2. **断行错误**：将被 OCR 错误切断的句子重新拼合为完整段落
3. **段落与章节结构**：恢复正确的段落分隔

### 绝对不得修改的内容
- 所有 LaTeX 公式：$...$、$$...$$、\\[...\\]、\\(...\\)
- 所有化学式
- 所有 Markdown 语法：标题(#)、**粗体**、*斜体*、列表、表格(|)、代码块(```)、链接、图片
- 所有代码块内容
- 书中原有的专业术语、人名、地名

### 不确定性标记
若某段文字极度模糊、无法读通，或逻辑完全断裂，在该段末尾插入：<unclear/>

## 重要提醒
- 校对后的文本必须完整，不可省略或截断
- 如果原文没有问题，原样返回即可（不要强行修改）
- 保持原文风格和语气
"""

# 仅校对模式的系统提示词（跳过转换/切分步骤）
AGENT_PROOFREAD_ONLY_PROMPT = """\
你是 pdf2md Agent——一个专业的 PDF 书籍转 Markdown 助手。
你可以调用工具完成校对工作，并用你自己的语言能力直接完成校对。

## 工作流程

请严格按以下步骤执行（不要跳步）：

### 第一步：加载章节
调用 `load_existing_chapters` 工具，从已有目录加载章节文件。

### 第二步：生成校对计划
调用 `write_plan` 工具生成 plan.md。

### 第三步：逐章逐段校对
（同上，对每一章逐段校对）
1. 调用 `get_chapter_segment` 获取一个段落及其上下文
2. 仔细阅读段落内容，按照【校对规则】进行修正
3. 调用 `save_proofread_segment`，将完整的校对后段落作为 corrected_text 传入
4. 重复以上 1-3 步，直到该章所有段落校对完毕
5. 调用 `finalize_chapter` 合并并保存该章的最终文件
6. 继续下一章，中途不停

### 第四步：完成
调用 `complete` 工具并附上处理摘要。

## 校对规则

### 必须修正的内容
1. **OCR 识别错误**：
   - 字母/数字混淆："1"↔"l"/"I"、"0"↔"O"、"rn"↔"m"、"cl"↔"d"
   - 中文形近字/同音字误识别（结合上下文语义判断）
   - 标点符号错误（全半角混用、引号不匹配）
2. **断行错误**：将被 OCR 错误切断的句子重新拼合为完整段落
3. **段落与章节结构**：恢复正确的段落分隔

### 绝对不得修改的内容
- 所有 LaTeX 公式：$...$、$$...$$、\\[...\\]、\\(...\\)
- 所有化学式
- 所有 Markdown 语法：标题(#)、**粗体**、*斜体*、列表、表格(|)、代码块(```)、链接、图片
- 所有代码块内容
- 书中原有的专业术语、人名、地名

### 不确定性标记
若某段文字极度模糊、无法读通，或逻辑完全断裂，在该段末尾插入：<unclear/>

## 重要提醒
- 校对后的文本必须完整，不可省略或截断
- 如果原文没有问题，原样返回即可
- 保持原文风格和语气
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent 类
# ─────────────────────────────────────────────────────────────────────────────

class PDF2MDAgent:
    """
    面向 Agent 的 PDF → Markdown 转换器。

    使用 OpenAI function calling 驱动工具调用，
    Agent（LLM）自身完成校对，无需独立的校对模块。
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        self._api_key = api_key or cfg.llm_api_key
        self._base_url = base_url or cfg.llm_base_url
        self._model = model or cfg.llm_model
        self._on_status = on_status or (lambda _msg: None)

        if not self._api_key:
            raise ValueError("未设置 OPENAI_API_KEY，请在 .env 中配置")

        self.client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        self.messages: list[dict] = []
        self.ctx: ToolContext | None = None

    # ─── 主入口 ────────────────────────────────────────────────

    def run(
        self,
        pdf_path: str,
        output_dir: str,
        book_name: str = "",
        converter: str = "marker",
    ) -> dict:
        """
        执行完整的 PDF → Markdown → 校对 流程。

        返回包含输出目录、迭代次数、耗时等信息的字典。
        """
        self.ctx = ToolContext(output_dir)

        self.messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请将以下 PDF 书籍转换为高质量 Markdown 并进行全量校对。\n\n"
                    f"- PDF 文件路径：{pdf_path}\n"
                    f"- 输出目录：{output_dir}\n"
                    f"- 书名：{book_name or '（从文件名推断）'}\n"
                    f"- 转换器：{converter}\n\n"
                    f"请开始执行。"
                ),
            },
        ]

        return self._agent_loop()

    def run_proofread_only(
        self,
        chapters_dir: str,
        output_dir: str,
    ) -> dict:
        """
        仅对已有章节目录进行校对（跳过 PDF 转换与切分）。

        返回包含输出目录、迭代次数、耗时等信息的字典。
        """
        self.ctx = ToolContext(output_dir)

        self.messages = [
            {"role": "system", "content": AGENT_PROOFREAD_ONLY_PROMPT},
            {
                "role": "user",
                "content": (
                    f"请对以下目录中的章节文件进行全量校对。\n\n"
                    f"- 章节目录：{chapters_dir}\n"
                    f"- 输出目录：{output_dir}\n\n"
                    f"请开始执行。"
                ),
            },
        ]

        return self._agent_loop()

    # ─── Agent 循环 ────────────────────────────────────────────

    def _agent_loop(self) -> dict:
        """核心 agent 循环：发送请求 → 处理工具调用 → 重复，直到任务完成。"""
        t_start = time.time()
        max_iterations = 500
        iteration = 0

        self._on_status("Agent 开始执行…")

        while iteration < max_iterations and not self.ctx.completed:
            iteration += 1

            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    temperature=0.2,
                )
            except Exception as e:
                logger.error("LLM API 调用失败：%s", e)
                self._on_status(f"API 错误：{e}，2 秒后重试…")
                time.sleep(2)
                continue

            choice = response.choices[0]
            message = choice.message

            # 将 assistant 消息加入历史
            self.messages.append(_serialize_message(message))

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    self._on_status(f"调用工具：{fn_name}")
                    logger.info("Agent 调用工具：%s(%s)", fn_name, json.dumps(fn_args, ensure_ascii=False)[:200])

                    result = execute_tool(self.ctx, fn_name, fn_args)
                    logger.debug("工具结果：%s", result[:300] if len(result) > 300 else result)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                    # 特殊处理：finalize_chapter 后裁剪上下文
                    if fn_name == "finalize_chapter":
                        self._checkpoint_after_chapter()

                # 章内消息数过多时裁剪
                self._trim_if_needed()

            elif choice.finish_reason == "stop":
                if message.content:
                    logger.info("Agent: %s", message.content[:200])
                    self._on_status(message.content[:100])

        elapsed = time.time() - t_start

        if iteration >= max_iterations:
            logger.warning("Agent 达到最大迭代次数 %d，强制停止", max_iterations)

        self._on_status(f"Agent 完成，共 {iteration} 轮，耗时 {elapsed:.1f}s")

        return {
            "output_dir": str(self.ctx.output_dir),
            "iterations": iteration,
            "elapsed_sec": round(elapsed, 1),
            "completed": self.ctx.completed,
        }

    # ─── 上下文管理 ────────────────────────────────────────────

    def _checkpoint_after_chapter(self) -> None:
        """
        一章校对完成后，重置对话历史以释放上下文空间。
        保留：系统提示 + 进度摘要 + 继续指令。
        """
        progress = self._build_progress_summary()
        system_msg = self.messages[0]  # 系统提示词不变

        self.messages = [
            system_msg,
            {
                "role": "user",
                "content": (
                    f"{progress}\n\n"
                    f"请继续校对下一章。如果所有章节已完成，请调用 complete 工具。"
                ),
            },
        ]
        logger.debug("对话历史已重置（章节检查点），当前 %d 条消息", len(self.messages))

    def _trim_if_needed(self) -> None:
        """章内消息过多时进行裁剪，保留系统提示 + 初始用户消息 + 最近消息。"""
        if len(self.messages) <= _MAX_MESSAGES_PER_CHAPTER:
            return

        system_msg = self.messages[0]
        user_msg = self.messages[1]
        recent = self.messages[-(_MAX_MESSAGES_PER_CHAPTER - 2):]

        # 避免裁剪位置正好在 tool 消息上（tool 消息必须跟在 assistant 后面）
        start = 0
        while start < len(recent) and recent[start].get("role") == "tool":
            start += 1

        self.messages = [system_msg, user_msg] + recent[start:]
        logger.debug("对话历史已裁剪至 %d 条消息", len(self.messages))

    def _build_progress_summary(self) -> str:
        """构建当前校对进度摘要，供上下文重置时使用。"""
        lines = ["【当前进度】"]
        for ch in self.ctx.chapters:
            segs = self.ctx.chapter_segments.get(ch.filename, [])
            done = self.ctx.proofread_segments.get(ch.filename, {})
            total = len(segs)
            proofread_cnt = len(done)
            status = "✓ 已完成" if proofread_cnt == total and total > 0 else f"待校对"
            lines.append(f"  [{ch.index:02d}] {ch.title} — {status}（{proofread_cnt}/{total} 段）")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _serialize_message(message) -> dict:
    """将 OpenAI ChatCompletionMessage 序列化为可重复使用的 dict。"""
    msg = {
        "role": "assistant",
        "content": message.content,
    }
    if message.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in message.tool_calls
        ]
    return msg


# ─────────────────────────────────────────────────────────────────────────────
# 便捷入口
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_pipeline(
    pdf_path: str,
    output_dir: str,
    book_name: str = "",
    converter: str = "marker",
    *,
    model: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict:
    """
    便捷函数：通过 agent 执行完整 PDF → Markdown 流程。

    参数
    ----
    pdf_path    : PDF 文件路径
    output_dir  : 输出目录
    book_name   : 书名（为空则从文件名推断）
    converter   : 转换后端（marker / magic-pdf）
    model       : LLM 模型名称（覆盖 .env 配置）
    on_status   : 状态回调函数
    """
    agent = PDF2MDAgent(model=model, on_status=on_status)
    return agent.run(pdf_path, output_dir, book_name, converter)


def run_agent_proofread(
    chapters_dir: str,
    output_dir: str,
    *,
    model: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict:
    """
    便捷函数：通过 agent 仅对已有章节进行校对。

    参数
    ----
    chapters_dir : 已有章节文件目录
    output_dir   : 输出目录
    model        : LLM 模型名称
    on_status    : 状态回调函数
    """
    agent = PDF2MDAgent(model=model, on_status=on_status)
    return agent.run_proofread_only(chapters_dir, output_dir)
