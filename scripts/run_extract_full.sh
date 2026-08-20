#!/usr/bin/env bash
# 遇到错误、未定义变量或管道失败时立刻退出，避免长任务悄悄写出不完整结果。
set -euo pipefail

# 默认从脚本位置定位仓库；可通过环境变量覆盖。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"

# 进入项目目录，保证后续相对路径都从项目根目录开始解析。
cd "$PROJECT_DIR"
# 创建日志目录和全量输出目录；目录存在时不报错。
mkdir -p logs outputs/demo_full

# 下载端点和离线模式均可由调用者按环境设置。
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

# 后台启动全量向量抽取任务。
nohup "$PYTHON" -u src/extract_embeddings.py \
  --items data/processed/demo/items.jsonl \
  --project_root . \
  --output_dir outputs/demo_full \
  --text_batch_size 512 \
  --image_batch_size 128 \
  --resume \
  --dedupe_images \
  > logs/extract_full.log 2>&1 &

# 保存后台任务进程号，便于后续检查或停止任务。
echo "$!" > outputs/demo_full/extract.pid
# 输出启动成功提示和进程号。
echo "已启动 extract_embeddings.py，进程号=$(cat outputs/demo_full/extract.pid)"
# 输出日志路径，方便远程查看任务进度。
echo "日志路径：$PROJECT_DIR/logs/extract_full.log"
