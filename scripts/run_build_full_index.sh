#!/usr/bin/env bash
# 遇到错误、未定义变量或管道失败时立刻退出，避免继续生成不可靠结果。
set -euo pipefail

# 默认从脚本位置定位仓库；可通过环境变量覆盖。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON="${PYTHON:-python}"

# 进入项目目录，保证后续相对路径都从项目根目录开始解析。
cd "$PROJECT_DIR"
# 创建日志目录和全量输出目录；目录存在时不报错。
mkdir -p logs outputs/demo_full

# 后台启动全量索引构建任务。
nohup "$PYTHON" -u src/build_faiss_index.py \
  --embedding_dir outputs/demo_full \
  --chunk_size 50000 \
  --hnsw_m 32 \
  --ef_construction 200 \
  --replace_existing \
  > logs/build_full_index.log 2>&1 &

# 保存后台任务进程号，便于后续检查任务状态。
echo "$!" > outputs/demo_full/build_index.pid
# 输出启动成功提示和进程号。
echo "已启动 build_faiss_index.py，进程号=$(cat outputs/demo_full/build_index.pid)"
# 输出日志路径，方便远程查看任务进度。
echo "日志路径：$PROJECT_DIR/logs/build_full_index.log"
