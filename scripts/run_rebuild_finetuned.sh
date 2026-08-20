#!/usr/bin/env bash
# 使用导出的微调模型重新抽取全量向量并构建新索引；绝不写入 outputs/demo_full。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/outputs/finetune_lora/exported_model}"
OUTPUT_DIR="${FINETUNED_INDEX_DIR:-$PROJECT_DIR/outputs/finetuned_full}"
ITEMS="${ITEMS:-$PROJECT_DIR/data/processed/demo/items.jsonl}"
JOB_DIR="$PROJECT_DIR/outputs/finetune_lora"
LOG_FILE="$JOB_DIR/rebuild.log"
PID_FILE="$JOB_DIR/rebuild.pid"
EXIT_CODE_FILE="$JOB_DIR/rebuild_exit_code"

cd "$PROJECT_DIR"

# 规范化路径，连带阻止用 .. 或重复斜杠绕过基线目录保护。
MODEL_DIR="$(realpath -m "$MODEL_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
BASELINE_DIR="$(realpath -m "$PROJECT_DIR/outputs/demo_full")"

# 明确阻止任何变量误配覆盖原始 demo_full 基线。
if [[ "$OUTPUT_DIR" == "$BASELINE_DIR" ]]; then
  echo "拒绝写入基线目录：$OUTPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/export_manifest.json" ]]; then
  echo "找不到已验证的微调模型：$MODEL_DIR/export_manifest.json" >&2
  echo "请先运行：bash scripts/run_finetune_export.sh" >&2
  exit 1
fi
if [[ ! -f "$ITEMS" ]]; then
  echo "找不到全量 item 清单：$ITEMS" >&2
  exit 1
fi

# 任务日志不能放进新的向量目录，否则抽取器会把它识别为不明旧产物并拒绝启动。
mkdir -p "$JOB_DIR" "$OUTPUT_DIR"
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "重建任务已经在运行，进程号：$EXISTING_PID"
    echo "查看日志：tail -f $LOG_FILE"
    exit 1
  fi
fi

rm -f "$EXIT_CODE_FILE"
nohup bash -c '
  exit_code_file="$1"
  project_dir="$2"
  python_bin="$3"
  model_dir="$4"
  output_dir="$5"
  items="$6"
  cd "$project_dir" || exit 1
  set +e
  "$python_bin" -u src/extract_embeddings.py \
    --items "$items" \
    --project_root "$project_dir" \
    --output_dir "$output_dir" \
    --model_name "$model_dir" \
    --text_batch_size 512 \
    --image_batch_size 128 \
    --max_length 52 \
    --device auto \
    --resume \
    --dedupe_images
  code=$?
  if [[ "$code" -eq 0 ]]; then
    "$python_bin" -u src/build_faiss_index.py \
      --embedding_dir "$output_dir" \
      --output_dir "$output_dir" \
      --chunk_size 50000 \
      --hnsw_m 32 \
      --ef_construction 200 \
      --replace_existing
    code=$?
  fi
  printf "%s\n" "$code" > "$exit_code_file"
  exit "$code"
' _ "$EXIT_CODE_FILE" "$PROJECT_DIR" "$PYTHON" "$MODEL_DIR" "$OUTPUT_DIR" "$ITEMS" \
  > "$LOG_FILE" 2>&1 &

REBUILD_PID="$!"
printf "%s\n" "$REBUILD_PID" > "$PID_FILE"
echo "微调模型全量向量与索引重建已在后台启动，进程号：$REBUILD_PID"
echo "新输出目录：$OUTPUT_DIR"
echo "查看日志：tail -f $LOG_FILE"
echo "基线目录保持不变：$PROJECT_DIR/outputs/demo_full"
