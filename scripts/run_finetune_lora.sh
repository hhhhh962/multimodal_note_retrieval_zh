#!/usr/bin/env bash
# 在远程服务器后台启动 Chinese-CLIP LoRA 微调，并持久保存日志、进程号和退出码。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/finetune_lora.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/finetune_lora}"
LOG_FILE="$OUTPUT_DIR/train.log"
PID_FILE="$OUTPUT_DIR/train.pid"
EXIT_CODE_FILE="$OUTPUT_DIR/exit_code"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

# 已有训练进程仍存活时拒绝重复启动，避免两次任务同时写同一套 checkpoint。
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "训练已经在运行，进程号：$EXISTING_PID"
    echo "查看进度：bash scripts/show_finetune_progress.sh"
    exit 1
  fi
fi

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "找不到 Python：$PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "找不到训练配置：$CONFIG" >&2
  exit 1
fi

# 可直接追加 --resume 或 --resume /path/to/checkpoint；参数原样交给训练入口。
COMMAND=("$PYTHON" -u -m src.finetune.train --config "$CONFIG")
if [[ "$#" -gt 0 ]]; then
  COMMAND+=("$@")
fi

# 新任务启动前移走旧退出码；训练完成或失败后，后台包装器都会写回真实退出码。
rm -f "$EXIT_CODE_FILE"
nohup bash -c '
  exit_code_file="$1"
  shift
  set +e
  "$@"
  code=$?
  printf "%s\n" "$code" > "$exit_code_file"
  exit "$code"
' _ "$EXIT_CODE_FILE" "${COMMAND[@]}" > "$LOG_FILE" 2>&1 &

TRAIN_PID="$!"
printf "%s\n" "$TRAIN_PID" > "$PID_FILE"

echo "LoRA 微调已在后台启动，进程号：$TRAIN_PID"
echo "训练日志：$LOG_FILE"
echo "进度快照：$OUTPUT_DIR/progress.json"
echo "查看进度：bash scripts/show_finetune_progress.sh"
echo "持续刷新：watch -n 10 bash scripts/show_finetune_progress.sh"
echo "请等待远程训练完成；此脚本不会自动执行评估、导出或索引重建。"
