# 中文多模态检索系统

基于 Chinese-CLIP、LoRA 和 FAISS HNSW 的中文图文检索项目，支持文本搜图、图片搜图和图文联合检索，并提供数据准备、向量抽取、索引构建、评估、微调、模型导出与 Gradio 演示的完整代码路径。

> 本仓库只发布源码、配置和测试。数据集、模型权重、embedding、FAISS 索引及评估产物不在仓库中，因此克隆后需要先准备数据和模型产物，不能直接启动在线 Demo。

## 项目做了什么

- 使用 `OFA-Sys/chinese-clip-vit-base-patch16` 编码中文文本与图片；
- 同时维护文本向量索引和图片向量索引；
- 文本、图片或图文联合查询最多形成四路召回，再做 late fusion；
- 使用归一化向量和 `IndexHNSWFlat` 内积索引；
- 对 MUGE 的“一段文本对应多张图片”结果做文档级聚合与去重；
- 支持 Chinese-CLIP 双塔 LoRA 微调、断点恢复、验证、导出和索引重建；
- 提供命令行检索、人工相关性评估、HNSW 近似召回评估和 Gradio 页面。

当前在线检索链路使用四路召回与分数融合。`src/rerank_bge.py` 是独立实验模块，尚未接入 `SearchPipeline`，README 不把它计入当前线上流程。

## 检索架构

```text
文本查询 ── Chinese-CLIP text encoder ─┬─> text HNSW ─┐
                                      └─> image HNSW ─┤
                                                       ├─> 分数融合
图片查询 ── Chinese-CLIP image encoder ┬─> text HNSW ─┤    ─> MUGE 聚合/去重
                                      └─> image HNSW ─┘    ─> Top-K
```

- Chinese-CLIP 查询编码可使用 GPU；
- HNSW 索引保持在 CPU，避免错误地把 HNSW 描述成 GPU FAISS；
- 默认索引参数为 `M=32`、`efConstruction=200`、`efSearch=512`；
- 模型、embedding 元数据和索引元数据在启动时进行一致性校验，防止错配。

## 已验证结果

以下数字来自本项目服务器上保留的固定评估产物，本次 README 更新时已重新核对。仓库未包含数据、权重和报告文件，因此这些数字表示一次已完成运行的结果快照，不等同于克隆仓库即可复现。

### LoRA 微调验证集

五项宏平均覆盖 MUGE 文搜图、Flickr30k-CN 双向检索和 COCO-CN 双向检索：

| 模型 | Five-task macro mean |
| --- | ---: |
| Chinese-CLIP 基线 | 0.7759 |
| 最佳 LoRA checkpoint | 0.8556 |
| 绝对提升 | +0.0798 |

最佳 checkpoint 的分任务结果：

| 任务 | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| MUGE text → image | 0.5633 | 0.8057 | 0.8778 |
| Flickr30k-CN text → image | 0.6810 | 0.9028 | 0.9426 |
| Flickr30k-CN image → text | 0.8520 | 0.9710 | 0.9930 |
| COCO-CN text → image | 0.7073 | 0.9273 | 0.9755 |
| COCO-CN image → text | 0.7240 | 0.9310 | 0.9800 |

### 索引与人工评估

| 评估 | 指标 | 结果 |
| --- | --- | ---: |
| HNSW 对内存 `IndexFlatIP` 的最终结果保持率 | Recall@5 | 0.9874 |
| HNSW 对内存 `IndexFlatIP` 的最终结果保持率 | Recall@10 | 0.9882 |
| 30 条人工查询逐图标注 | Precision@5 | 0.9733 |
| 30 条人工查询逐图标注 | Precision@10 | 0.9333 |
| 30 条人工查询逐图标注 | NDCG@5 | 0.9567 |
| 30 条人工查询逐图标注 | NDCG@10 | 0.9456 |

这里的 HNSW Recall 衡量“近似索引是否保持 Flat 检索结果”，不是模型本身的语义检索准确率；人工指标的样本量只有 30 条，应视为小规模质量检查。

## 目录结构

```text
.
├── app.py                         # Gradio 文本/图片/联合检索页面
├── configs/
│   └── finetune_lora.yaml         # LoRA 训练与验证配置
├── scripts/
│   ├── prepare_retrieval_datasets.py
│   ├── run_finetune_lora.sh
│   ├── run_finetune_eval.sh
│   ├── run_finetune_export.sh
│   └── run_rebuild_finetuned.sh
├── src/
│   ├── prepare_demo_items.py      # 统一样本与图片准备
│   ├── extract_embeddings.py      # 图文向量抽取
│   ├── build_faiss_index.py       # HNSW 索引构建与校验
│   ├── search_pipeline.py         # 四路召回、融合与聚合
│   ├── search_faiss.py            # 命令行检索
│   ├── evaluate.py                # 人工标注评估
│   ├── hnsw_final_benchmark.py    # HNSW 对 Flat 的最终 Recall
│   └── finetune/                  # 训练、验证、checkpoint 与导出
├── tests/
│   └── test_hnsw_index.py
├── FINETUNE.md                    # 微调流程细节
├── HANDOFF.md                     # 运行入口速查
└── NOTICE.md                      # 第三方资源与授权边界
```

## 环境准备

参考运行环境为 Linux、Python 3.10 和 CUDA 12。`requirements.txt` 默认包含 GPU 版 FAISS，CPU 或其他 CUDA 环境需要自行替换对应的 FAISS 包。

```bash
git clone https://github.com/hhhhh962/project_1_multimodal_search.git
cd project_1_multimodal_search

conda create -n search python=3.10 -y
conda activate search
pip install -r requirements.txt
```

如需把 Hugging Face 和 pip 缓存放到其他磁盘：

```bash
export CACHE_ROOT="/path/to/cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
```

## 数据格式

检索流水线读取 `items.jsonl`，每行对应一个文本—图片对：

```json
{
  "item_id": "MUGE_train_10001_20001",
  "text": "一段中文描述",
  "image_id": "20001",
  "image_path": "data/images/demo/MUGE/train/20001.jpg",
  "source": "MUGE",
  "split": "train",
  "text_id": "10001"
}
```

仓库支持 MUGE、Flickr30k-CN 和 COCO-CN，但不分发这些数据。请自行获取并遵守各数据集的访问和使用条款。默认微调配置期望目录为：

```text
data/
├── MUGE/
├── Flickr30k-CN/
└── COCO-CN/
```

如数据不在仓库目录中，请复制并修改 `configs/finetune_lora.yaml`，不要提交机器专属绝对路径。

## 从数据到检索

### 1. 准备统一样本

原始 LMDB 已符合 `src/prepare_demo_items.py` 预期结构时：

```bash
python src/prepare_demo_items.py \
  --datasets-root /path/to/datasets \
  --project-root . \
  --output-name demo
```

输出包括 `data/processed/demo/items.jsonl`、解码后的引用图片和人工查询模板。

### 2. 抽取图文向量

```bash
python src/extract_embeddings.py \
  --items data/processed/demo/items.jsonl \
  --project_root . \
  --output_dir outputs/demo_full \
  --model_name OFA-Sys/chinese-clip-vit-base-patch16 \
  --device auto \
  --resume \
  --dedupe_images
```

### 3. 构建 HNSW 索引

```bash
python src/build_faiss_index.py \
  --embedding_dir outputs/demo_full \
  --hnsw_m 32 \
  --ef_construction 200 \
  --replace_existing
```

构建器会校验向量数量、维度、L2 归一化、模型指纹和数据清单，并写出 `text.index`、`image.index` 与 `index_meta.json`。

### 4. 命令行检索

文本查询：

```bash
python src/search_faiss.py \
  --embedding_dir outputs/demo_full \
  --model_name OFA-Sys/chinese-clip-vit-base-patch16 \
  --query "圣诞节沙发靠垫" \
  --top_k_final 10
```

图片或图文联合查询可增加 `--image_path /path/to/query.jpg`。至少提供文本或图片中的一个。

### 5. 启动 Gradio

`app.py` 默认读取微调后的模型和索引。使用基线产物时显式覆盖：

```bash
MODEL_NAME=OFA-Sys/chinese-clip-vit-base-patch16 \
EMBEDDING_DIR=outputs/demo_full \
DEVICE=auto \
python app.py
```

页面默认监听 `0.0.0.0:7860`。模型与索引必须来自同一套产物，否则启动时的一致性校验会拒绝加载。

## LoRA 微调

微调覆盖训练、断点恢复、五任务验证、最佳 checkpoint 选择、模型合并导出和全量索引重建。完整参数与产物说明见 [FINETUNE.md](FINETUNE.md)。

```bash
bash scripts/run_finetune_lora.sh
bash scripts/show_finetune_progress.sh
bash scripts/run_finetune_eval.sh valid
bash scripts/run_finetune_export.sh
bash scripts/run_rebuild_finetuned.sh
```

脚本自动定位仓库根目录，并允许通过 `PYTHON`、`CONFIG`、`OUTPUT_DIR`、`MODEL_DIR` 等环境变量覆盖默认值。

## 评估与测试

HNSW 近似召回质量：

```bash
python src/hnsw_final_benchmark.py \
  --embedding_dir outputs/finetuned_full \
  --sample_size 1000 \
  --candidate_pool_size 100
```

轻量回归测试不需要完整数据或模型：

```bash
python -m unittest discover -s tests -v
```

当前公开提交包含 11 项测试，覆盖 HNSW 构建/加载、参数校验、元数据错配、CPU HNSW 行为、最终融合、MUGE 去重和 Recall 计算。

## 已知限制

- 仓库不包含可直接运行的模型、索引、图片或数据集；
- 当前在线 `SearchPipeline` 未接入 BGE reranker；
- HNSW 运行在 CPU，GPU 主要用于 Chinese-CLIP 编码；
- 人工评估只有 30 条查询，不能替代更大规模的盲测；
- 默认依赖面向 Linux/CUDA 12，其他平台需要调整 PyTorch 与 FAISS 安装方式；
- 本仓库暂未授予开源许可证，第三方模型和数据另受各自条款约束，详见 [NOTICE.md](NOTICE.md)。
