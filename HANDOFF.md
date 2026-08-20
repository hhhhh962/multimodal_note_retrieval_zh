# 项目交接说明

本仓库包含中文多模态检索系统的源码、配置、启动脚本和轻量测试。数据集、模型权重、向量、FAISS 索引、日志与评估产物不在源码仓库中。

## 快速开始

```bash
conda create -n search python=3.10 -y
conda activate search
pip install -r requirements.txt
```

准备数据时，将三套数据分别放到：

```text
data/MUGE/
data/Flickr30k-CN/
data/COCO-CN/
```

默认训练配置为 `configs/finetune_lora.yaml`。如果数据或输出位于其他磁盘，请复制配置文件并修改路径，再通过 `CONFIG` 环境变量传给脚本：

```bash
CONFIG=/path/to/finetune_lora.yaml bash scripts/run_finetune_lora.sh
```

## 工作流

1. 使用 `scripts/prepare_retrieval_datasets.py` 准备检索数据。
2. 使用 `scripts/run_finetune_lora.sh` 启动 LoRA 微调。
3. 使用 `scripts/show_finetune_progress.sh` 查看进度。
4. 使用 `scripts/run_finetune_eval.sh` 评估最佳 checkpoint。
5. 使用 `scripts/run_finetune_export.sh` 导出合并模型。
6. 使用 `scripts/run_rebuild_finetuned.sh` 重建向量与索引。

脚本默认自动定位仓库根目录并使用当前环境的 `python`。可通过 `PROJECT_DIR`、`PYTHON`、`CONFIG`、`OUTPUT_DIR` 等环境变量覆盖。

## 验证

```bash
python -m unittest discover -s tests -v
```

轻量测试不需要下载完整数据集或模型权重。

## 产物边界

以下内容由 `.gitignore` 排除，不应提交到源码分支：

- 原始或处理后的数据；
- 模型 checkpoint 与导出权重；
- NumPy 向量和 FAISS 索引；
- 训练日志、PID、评估报告及缓存。

第三方模型和数据集的使用条件见 `NOTICE.md`。
