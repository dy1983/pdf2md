#!/usr/bin/env python3
"""
main.py
~~~~~~~
命令行入口。

用法示例
--------
# 完整流程（转换 + 切分 + 校对）
python main.py convert book.pdf

# 仅转换 + 切分，不调用 LLM
python main.py convert book.pdf --skip-proofread

# 仅对已有章节目录进行 LLM 校对
python main.py proofread output/MyBook/chapters/

# 使用指定模型和自定义输出根目录
python main.py convert book.pdf --model gpt-4o --output results/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich import print as rprint

from config import cfg
from src.pipeline import PipelineResult, ChapterSummary, run_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    # 屏蔽第三方库的冗余日志
    for noisy in ("httpx", "httpcore", "openai", "PIL", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# 结果展示
# ─────────────────────────────────────────────────────────────────────────────

def _print_result(result: PipelineResult) -> None:
    if result.error:
        console.print(f"[bold red]✗ 错误：{result.error}[/bold red]")
        return

    # 章节详情表格
    table = Table(title=f"《{result.book_name}》转换结果", show_lines=True)
    table.add_column("序号", style="cyan", justify="right")
    table.add_column("章节标题", style="white")
    table.add_column("字符数", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("置信度", justify="right")
    table.add_column("<unclear/>", justify="right")
    table.add_column("耗时(s)", justify="right")

    for s in result.chapter_summaries:
        conf_color = (
            "green" if s.confidence >= 85
            else "yellow" if s.confidence >= 65
            else "red"
        )
        table.add_row(
            f"{s.index:02d}",
            s.title[:40],
            f"{s.char_count:,}",
            str(s.chunk_count) if not s.skipped else "-",
            f"[{conf_color}]{s.confidence}%[/{conf_color}]",
            str(s.unclear_cnt) if s.unclear_cnt else "-",
            f"{s.elapsed_sec:.1f}" if not s.skipped else "-",
        )

    console.print(table)

    total_unclear = sum(s.unclear_cnt for s in result.chapter_summaries)
    avg_conf = (
        round(sum(s.confidence for s in result.chapter_summaries) / len(result.chapter_summaries))
        if result.chapter_summaries
        else 0
    )

    console.print(
        Panel(
            f"[bold]输出目录[/bold]：{result.output_dir}\n"
            f"[bold]章节总数[/bold]：{result.total_chapters}\n"
            f"[bold]全书置信度[/bold]：{avg_conf}%\n"
            f"[bold]<unclear/> 总计[/bold]：{total_unclear} 处\n"
            f"[bold]转换耗时[/bold]：{result.convert_elapsed:.1f}s\n"
            f"[bold]总耗时[/bold]：{result.total_elapsed:.1f}s",
            title="[green]✓ 完成[/green]",
            border_style="green",
        )
    )


# ─────────────────────────────────────────────────────────────────────────────
# 子命令处理
# ─────────────────────────────────────────────────────────────────────────────

def cmd_convert(args: argparse.Namespace) -> int:
    pdf = Path(args.pdf)
    if not pdf.exists():
        console.print(f"[red]错误：找不到文件 {pdf}[/red]")
        return 1

    # 覆盖 config（命令行 > 环境变量）
    if args.model:
        cfg.llm_model = args.model
    if args.converter:
        cfg.converter = args.converter

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )

    chapter_task = None
    total_ref = [0]

    with progress:
        convert_task = progress.add_task("PDF 转换中…", total=None)

        def on_chapter_start(index: int, total: int, title: str) -> None:
            nonlocal chapter_task
            total_ref[0] = total
            if chapter_task is None:
                progress.update(convert_task, completed=1, total=1)
                chapter_task = progress.add_task(
                    f"LLM 校对…", total=total
                )
            progress.update(
                chapter_task,
                description=f"[bold cyan]校对 [{index:02d}/{total}] {title[:30]}",
                completed=index,
            )

        def on_chapter_done(summary: ChapterSummary) -> None:
            if chapter_task is not None:
                progress.update(chapter_task, advance=1)

        result = run_pipeline(
            str(pdf),
            output_root=args.output,
            book_name=args.name,
            converter=args.converter,
            skip_proofread=args.skip_proofread,
            on_chapter_start=on_chapter_start,
            on_chapter_done=on_chapter_done,
        )

    _print_result(result)
    return 0 if not result.error else 1


def cmd_proofread(args: argparse.Namespace) -> int:
    chapters_dir = Path(args.chapters_dir)
    if not chapters_dir.exists():
        console.print(f"[red]错误：目录不存在 {chapters_dir}[/red]")
        return 1

    if args.model:
        cfg.llm_model = args.model

    # 推导 output_root 与 book_name：假设 chapters_dir 形如 output/<book>/chapters
    book_dir = chapters_dir.parent
    book_name = book_dir.name
    output_root = str(book_dir.parent)

    result = run_pipeline(
        pdf_path="",                  # 跳过转换
        output_root=output_root,
        book_name=book_name,
        proofread_only_dir=str(chapters_dir),
    )
    _print_result(result)
    return 0 if not result.error else 1


# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf2md",
        description="将书籍 PDF 转换为高质量 Markdown（保留公式、表格、图片）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── convert ──────────────────────────────────
    p_conv = sub.add_parser("convert", help="转换 PDF 并校对（完整流程）")
    p_conv.add_argument("pdf", help="输入 PDF 文件路径")
    p_conv.add_argument("-o", "--output", default="output", help="输出根目录（默认：output）")
    p_conv.add_argument("-n", "--name", default=None, help="书名（默认取 PDF 文件名）")
    p_conv.add_argument(
        "--converter",
        choices=["marker", "magic-pdf"],
        default=None,
        help="PDF 转换后端（默认：marker）",
    )
    p_conv.add_argument(
        "--model",
        default=None,
        help="LLM 模型名称（覆盖 .env 中 LLM_MODEL）",
    )
    p_conv.add_argument(
        "--skip-proofread",
        action="store_true",
        help="跳过 LLM 校对，仅输出原始转换结果",
    )
    p_conv.set_defaults(func=cmd_convert)

    # ── proofread ─────────────────────────────────
    p_proof = sub.add_parser("proofread", help="仅对已有章节目录进行 LLM 校对")
    p_proof.add_argument("chapters_dir", help="包含 .md 章节文件的目录路径")
    p_proof.add_argument(
        "--model",
        default=None,
        help="LLM 模型名称",
    )
    p_proof.set_defaults(func=cmd_proofread)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]已取消[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
