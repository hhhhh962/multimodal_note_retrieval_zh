#!/usr/bin/env bash
# 手工把最佳 LoRA checkpoint 合并为普通 Hugging Face Chinese-CLIP 模型。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$PROJECT_DIR/configs/finetune_lora.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/finetune_lora}"
BEST_POINTER="$OUTPUT_DIR/best_checkpoint.json"
CHECKPOINT_DIR="${1:-}"
EXPORT_DIR="${2:-$OUTPUT_DIR/exported_model}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"

cd "$PROJECT_DIR"

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
if [[ -d "$EXPORT_DIR" ]] && [[ -n "$(find "$EXPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "导出目录非空，拒绝覆盖：$EXPORT_DIR" >&2
  echo "请先换一个输出目录，或在确认后手工移走旧目录。" >&2
  exit 1
fi

echo "开始导出 checkpoint：$CHECKPOINT_DIR"
"$PYTHON" -u -m src.finetune.export_model \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT_DIR" \
  --output_dir "$EXPORT_DIR" \
  --device "$EXPORT_DEVICE"

echo "导出和双塔重载验证完成：$EXPORT_DIR"
