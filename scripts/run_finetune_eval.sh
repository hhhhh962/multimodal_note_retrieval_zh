#!/usr/bin/env bash
# 手工评估最佳 LoRA checkpoint；训练脚本不会自动调用本文件。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/finetune_lora.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/finetune_lora}"
BEST_POINTER="$OUTPUT_DIR/best_checkpoint.json"
SPLIT="${1:-valid}"
CHECKPOINT_DIR="${2:-}"

cd "$PROJECT_DIR"

if [[ "$SPLIT" != "valid" && "$SPLIT" != "test" ]]; then
  echo "用法：bash scripts/run_finetune_eval.sh [valid|test] [checkpoint目录]" >&2
  exit 2
fi

if [[ -z "$CHECKPOINT_DIR" ]]; then
  if [[ ! -f "$BEST_POINTER" ]]; then
    echo "找不到最佳 checkpoint 指针：$BEST_POINTER" >&2
    exit 1
  fi
  CHECKPOINT_DIR="$("$PYTHON" - "$BEST_POINTER" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["checkpoint_dir"])
PY
)"
fi

if [[ ! -d "$CHECKPOINT_DIR/adapter" ]]; then
  echo "checkpoint 缺少 adapter 目录：$CHECKPOINT_DIR" >&2
  exit 1
fi

REPORT_DIR="$OUTPUT_DIR/evaluation"
REPORT_FILE="$REPORT_DIR/best_${SPLIT}_metrics.json"
mkdir -p "$REPORT_DIR"

echo "开始评估：split=$SPLIT"
echo "checkpoint：$CHECKPOINT_DIR"
"$PYTHON" -u -m src.finetune.evaluate_retrieval \
  --config "$CONFIG" \
  --adapter "$CHECKPOINT_DIR" \
  --split "$SPLIT" \
  --device auto \
  --output "$REPORT_FILE"

echo "评估完成：$REPORT_FILE"
if [[ "$SPLIT" == "test" ]]; then
  echo "若 MUGE test 没有 GT，提交文件会写到：$REPORT_DIR/muge_test_predictions.jsonl"
fi
