#!/usr/bin/env bash
# 汇总 LoRA 微调的进程状态、progress.json、最新日志和当前 GPU 使用情况。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/finetune_lora}"
PROGRESS_FILE="$OUTPUT_DIR/progress.json"
LOG_FILE="$OUTPUT_DIR/train.log"
PID_FILE="$OUTPUT_DIR/train.pid"
EXIT_CODE_FILE="$OUTPUT_DIR/exit_code"
LOG_LINES="${LOG_LINES:-30}"

cd "$PROJECT_DIR"

echo "========== 微调进程 =========="
if [[ -f "$PID_FILE" ]]; then
  TRAIN_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$TRAIN_PID" =~ ^[0-9]+$ ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "状态：运行中"
    echo "进程号：$TRAIN_PID"
  else
    echo "状态：进程已结束"
    echo "记录的进程号：${TRAIN_PID:-无}"
  fi
else
  echo "状态：尚未找到 train.pid，训练可能还没有启动"
fi
if [[ -f "$EXIT_CODE_FILE" ]]; then
  EXIT_CODE="$(tr -d '[:space:]' < "$EXIT_CODE_FILE")"
  echo "退出码：$EXIT_CODE（0 表示正常完成，非 0 表示失败）"
else
  echo "退出码：尚未生成（运行中或尚未启动）"
fi

echo
echo "========== progress.json =========="
if [[ -f "$PROGRESS_FILE" ]]; then
  "$PYTHON" - "$PROGRESS_FILE" <<'PY'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    progress = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    print(f"暂时无法读取进度快照：{exc}")
    raise SystemExit(0)

def value(name, default="-"):
    current = progress.get(name)
    return default if current is None else current

def duration(seconds):
    if seconds is None:
        return "-"
    try:
        seconds = max(int(float(seconds)), 0)
    except (TypeError, ValueError):
        return str(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def gib(byte_count):
    if byte_count is None:
        return "-"
    try:
        return f"{float(byte_count) / (1024 ** 3):.2f} GiB"
    except (TypeError, ValueError):
        return str(byte_count)

step = progress.get("optimizer_step")
total = progress.get("total_optimizer_steps")
if isinstance(step, (int, float)) and isinstance(total, (int, float)) and total:
    percentage = min(max(float(step) / float(total) * 100.0, 0.0), 100.0)
    step_text = f"{int(step)}/{int(total)} ({percentage:.2f}%)"
else:
    step_text = f"{value('optimizer_step')}/{value('total_optimizer_steps')}"

loss = progress.get("loss")
loss_text = f"{float(loss):.6f}" if isinstance(loss, (int, float)) and math.isfinite(float(loss)) else str(value("loss"))
throughput = progress.get("throughput_samples_per_second")
throughput_text = f"{float(throughput):.2f} samples/s" if isinstance(throughput, (int, float)) else "-"

print(f"状态：{value('status')}（最新事件：{value('last_event')}）")
print(f"更新时间：{value('updated_at')}")
print(f"epoch：{value('epoch')}/{value('epochs')}")
print(f"优化步：{step_text}")
print(f"loss：{loss_text}")
print(f"精度 / micro-batch / 累计步数 / 有效 batch：{value('precision')} / {value('micro_batch_size')} / {value('accumulation_steps')} / {value('effective_batch_size')}")
print(f"吞吐：{throughput_text}")
print(f"已用时间 / 预计剩余：{duration(progress.get('elapsed_seconds'))} / {duration(progress.get('eta_seconds'))}")
print(f"最近验证分数 / 最佳分数：{value('validation_score')} / {value('best_score')}")
print(f"最佳 checkpoint：{value('best_checkpoint')}")
print(f"GPU 已分配 / 已保留 / 峰值 / 总显存：{gib(progress.get('gpu_memory_allocated_bytes'))} / {gib(progress.get('gpu_memory_reserved_bytes'))} / {gib(progress.get('gpu_memory_peak_bytes'))} / {gib(progress.get('gpu_memory_total_bytes'))}")
if progress.get("status") == "failed":
    print(f"失败类型：{value('error_type')}")
    print(f"失败原因：{value('error')}")
PY
else
  echo "尚未生成：$PROGRESS_FILE"
fi

echo
echo "========== 最新 $LOG_LINES 行日志 =========="
if [[ -f "$LOG_FILE" ]]; then
  tail -n "$LOG_LINES" "$LOG_FILE"
else
  echo "尚未生成：$LOG_FILE"
fi

echo
echo "========== 当前 GPU =========="
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader,nounits; then
    echo "nvidia-smi 当前不可用，服务器可能处于无卡模式。"
  fi
else
  echo "未找到 nvidia-smi，服务器可能处于无卡模式。"
fi
