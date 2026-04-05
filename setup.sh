#!/usr/bin/env bash
# setup.sh ─ 一键创建虚拟环境并安装依赖
# 用法：bash setup.sh
set -euo pipefail

MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
VENV=".venv"

echo "======================================================"
echo "  pdf2md 环境初始化"
echo "======================================================"

# ── 检查 Python ───────────────────────────────────────────
PY=$(command -v python3 || command -v python || true)
if [[ -z "$PY" ]]; then
    echo "[错误] 未找到 Python 3，请先安装 Python >= 3.10"
    exit 1
fi
PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[OK] Python $PY_VER → $PY"

# ── 创建虚拟环境 ──────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "[1/4] 创建虚拟环境 $VENV …"
    "$PY" -m venv "$VENV"
else
    echo "[1/4] 虚拟环境 $VENV 已存在，跳过创建"
fi

# 激活（兼容 bash/zsh）
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 升级基础工具 ──────────────────────────────────────────
echo "[2/4] 升级 pip / setuptools / wheel …"
pip install --quiet --upgrade pip setuptools wheel -i "$MIRROR"

# ── 安装主依赖 ────────────────────────────────────────────
echo "[3/4] 安装项目依赖（使用清华镜像）…"
pip install --quiet -r requirements.txt -i "$MIRROR" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── 配置文件初始化 ────────────────────────────────────────
echo "[4/4] 初始化配置文件 …"
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    echo "  已创建 .env（请编辑其中的 OPENAI_API_KEY 等参数）"
else
    echo "  .env 已存在，跳过"
fi

echo ""
echo "======================================================"
echo "  安装完成！"
echo ""
echo "  使用前请编辑 .env，填写您的 API Key。"
echo ""
echo "  示例命令："
echo "    source .venv/bin/activate"
echo "    python main.py convert 书籍.pdf"
echo "    python main.py convert 书籍.pdf --skip-proofread"
echo "    python main.py proofread output/书籍/chapters/"
echo "======================================================"
