# 中文多模态笔记检索

一个以“完整笔记”为检索单位的中文图文搜索项目。每篇笔记由一段文本和一组图片组成。

项目使用 Chinese-CLIP 编码文本与图片，使用两份 FAISS HNSW 索引召回候选，再对候选笔记的文本及全部图片精确重算四路相似度。仓库提供数据整理、向量抽取、旧产物迁移、索引构建、命令行检索、Gradio 页面、LoRA 微调和评估代码。

> 本仓库只包含源码、配置和测试，不包含数据集、模型权重、向量、FAISS 索引或评估产物。克隆后需要自行准备这些资源。

## 笔记样式

一段文字对应多张图片：

```text
笔记 A
├── 一段文本
├── 图片 1
├── 图片 2
└── 图片 3
```

某篇笔记的图片相关性取其全部图片中的最高分，但返回结果会包含这篇笔记的完整图片列表。

## 检索流程

```text
文本查询 ─┬─> 文本 HNSW（每行一篇笔记） ─┐
          └─> 图片 HNSW（每行一张唯一图片） ─┤
                                               ├─> 映射到候选笔记
图片查询 ─┬─> 文本 HNSW ─────────────────────┤
          └─> 图片 HNSW ─────────────────────┘
                                                    │
                                                    v
                         用候选笔记的文本和全部图片精确重算
                                                    │
                                                    v
                                      late fusion + 唯一笔记 Top-K
```

四路精确分数分别是：

- `text → text`：查询文本与笔记文本；
- `text → image`：查询文本与该笔记所有图片的最高分；
- `image → text`：查询图片与笔记文本；
- `image → image`：查询图片与该笔记所有图片的最高分。

文本索引和图片索引使用 `IndexHNSWFlat`、内积、`M=32`、`efConstruction=200`、`efSearch=512`。两份索引的行数允许不同：前者等于笔记数，后者等于全局唯一图片数。HNSW 保持在 CPU，GPU 只用于 Chinese-CLIP 查询编码。

## 已验证的模型结果

下表来自项目既有服务器产物，是 LoRA 模型在五项验证任务上的固定结果快照。

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

旧 pair-level 索引的 HNSW 和 30 条人工评估结果不再作为当前文档级检索结构的指标。文档级 HNSW benchmark 会由 `src/hnsw_final_benchmark.py` 重新生成；人工指标需要按完整笔记重新标注。

## 数据格式

检索流水线读取 `documents.jsonl`，每行是一篇完整笔记：

```json
{
  "doc_id": "MUGE:train:10001",
  "text": "一段中文描述",
  "source": "MUGE",
  "split": "train",
  "text_id": "10001",
  "images": [
    {
      "image_id": "20001",
      "image_path": "data/images/demo/MUGE/train/20001.jpg"
    },
    {
      "image_id": "20002",
      "image_path": "data/images/demo/MUGE/train/20002.jpg"
    }
  ]
}
```

`doc_id` 默认由 `source:split:text_id` 组成。同一 `doc_id` 的文本必须完全一致，图片按输入顺序去重。同一图片可以属于多篇笔记，向量只保存一次，召回后会展开到全部所属笔记。

## 快速开始

参考环境为 Linux、Python 3.10 和 CUDA 12。`requirements.txt` 默认安装 GPU 版 FAISS；CPU 或其他 CUDA 版本请替换成匹配的 FAISS 包。

```bash
git clone https://github.com/hhhhh962/project_1_multimodal_search.git
cd project_1_multimodal_search

conda create -n search python=3.10 -y
conda activate search
pip install -r requirements.txt
```

如需把缓存、数据和输出放到其他磁盘：

```bash
export HF_HOME=/path/to/cache/huggingface
export HF_DATASETS_CACHE=/path/to/cache/huggingface/datasets
export PIP_CACHE_DIR=/path/to/cache/pip
export DOCUMENTS=/path/to/documents.jsonl
export OUTPUT_DIR=/path/to/outputs/docs_v2
```

### 1. 准备笔记清单

仓库支持 MUGE、Flickr30k-CN 和 COCO-CN，但不分发这些数据。自行获取并遵守各数据集条款后，可从原始 LMDB 生成统一清单：

```bash
python src/prepare_demo_items.py \
  --datasets-root /path/to/datasets \
  --project-root . \
  --output-name demo
```

主要输出是 `data/processed/demo/documents.jsonl`。脚本也会保留 `items.jsonl`，用于训练兼容和旧向量迁移；新检索运行时不读取它。

### 2. 抽取文档级向量

`--model_name` 需要指向本地完整 Hugging Face 模型目录。若只想使用基础 Chinese-CLIP，请先下载其 snapshot 到本地目录。

```bash
python src/extract_embeddings.py \
  --documents data/processed/demo/documents.jsonl \
  --project_root . \
  --output_dir outputs/demo_docs_v2 \
  --model_name /path/to/chinese-clip-model \
  --device auto \
  --resume
```

输出中，`text_embeddings.npy` 的行数等于笔记数，`image_embeddings.npy` 的行数等于唯一图片数。为避免全量数据在小内存机器上加载成数十万个 Python 对象，`retrieval_meta.json` 只保存 schema、计数、指纹、哈希和文件引用；完整文档与图片资产保存在带随机访问偏移表的 JSONL 中，双向图文关系保存在 CSR NumPy 数组中。检索时只解析召回到的候选笔记。

### 3. 构建两份 HNSW 索引

```bash
python src/build_faiss_index.py \
  --embedding_dir outputs/demo_docs_v2 \
  --hnsw_m 32 \
  --ef_construction 200
```

构建器会校验 schema、数量、维度、L2 归一化、模型指纹和数据清单哈希，然后写出 `text.index`、`image.index` 和 `index_meta.json`。

内存只有约 2GB 时可增加 `--low_memory_build --faiss_threads 1 --chunk_size 5000`。该模式仅在建图阶段用 FP16 scalar-quantized 存储降低峰值，随后把邻接图装配回完整 float32 `IndexHNSWFlat`；`index_meta.json` 会同时记录建图存储与最终存储。应通过下文 benchmark 验证近似召回保持率。

如果文本建图已经完成转存、但在最终装配时进程中断，可在上述参数后增加 `--resume_text_spooled_graph`，复用输出目录中的 `.text_embeddings.hnsw_graph.tmp`；恢复成功后该临时目录会自动删除。

### 4. 检索

文本查询：

```bash
python src/search_faiss.py \
  --embedding_dir outputs/demo_docs_v2 \
  --model_name /path/to/chinese-clip-model \
  --query "圣诞节沙发靠垫" \
  --top_k_final 10
```

图片查询增加 `--image_path /path/to/query.jpg`；同时传入文本和图片就是联合查询。每条结果都包含 `image_paths`、`image_count`、`matched_image_path` 和四路分数。

### 5. 启动 Gradio

```bash
MODEL_DIR=/path/to/chinese-clip-model \
EMBEDDING_DIR=outputs/demo_docs_v2 \
DEVICE=auto \
python app.py
```

页面默认监听 `0.0.0.0:7860`。表格一篇笔记一行；图片墙按笔记排名连续展示每篇笔记的全部图片。

## 迁移旧 pair-level 产物

如果已经有一一对齐的旧 `text_embeddings.npy` 和 `image_embeddings.npy`，可复用向量，无需再次占用 GPU 编码：

```bash
python scripts/migrate_pair_artifacts.py \
  --source-dir outputs/finetuned_full \
  --items data/processed/demo/items.jsonl \
  --output-dir outputs/finetuned_docs_v2
```

迁移工具会先把复用向量在 float32 下重新做 L2 归一化，再默认以 `atol=1e-6` 验证同一笔记的重复文本向量、同一路径的重复图片向量是否一致，并拒绝覆盖任何已有目标目录。只有已经诊断出旧 FP16 分批编码漂移时，才应显式传入更大的 `--atol`；迁移报告会记录源范数范围、非完全相同行数和实际最大误差。旧目录保持不变。新代码遇到旧 schema 会直接提示运行该工具，不提供 pair-level 兼容分支。

迁移完成后再构建索引：

```bash
python src/build_faiss_index.py \
  --embedding_dir outputs/finetuned_docs_v2
```

## LoRA 微调

训练仍使用原有多正样本损失，一段文本对应多张图片时无需改成文档级 batch。训练、验证、模型合并导出和索引重建彼此独立：

```bash
bash scripts/run_finetune_lora.sh
bash scripts/show_finetune_progress.sh
bash scripts/run_finetune_eval.sh valid
bash scripts/run_finetune_export.sh
bash scripts/run_rebuild_finetuned.sh
```

脚本会自动定位仓库根目录，并允许通过 `PYTHON`、`CONFIG`、`MODEL_DIR`、`DOCUMENTS` 和 `FINETUNED_INDEX_DIR` 覆盖默认值。详细说明见 [FINETUNE.md](FINETUNE.md)。

## Benchmark 与测试

这个 benchmark 只检查 **FAISS HNSW 加速是否漏掉精确结果**，不评价搜索结果对人是否相关。它先把查询与全库 434,927 篇笔记逐一计算，得到精确 Top-K；再运行实际的 HNSW 候选召回与文档级重排，比较两份 Top-K 有多少篇相同。

```bash
python src/hnsw_final_benchmark.py \
  --embedding_dir outputs/finetuned_docs_v2 \
  --sample_size 100 \
  --candidate_pool_size 100
```

benchmark 默认以 10 篇为一批，分阶段加载文本和图片索引，再按固定随机排列汇总 100 篇结果；这样约 2 GB 内存的环境也不需要同时常驻两份索引。可通过 `--batch_size` 调整。

当前全量 v2 产物的固定样本结果（100 篇、候选池 100、`efSearch=512`）：

| 查询方式 | Recall@5 | Recall@10 |
| --- | ---: | ---: |
| 文本 | 1.000 | 1.000 |
| 图片 | 1.000 | 1.000 |
| 文本 + 图片 | 1.000 | 0.999 |
| 三种方式平均 | 1.000 | 0.9997 |

总体 Recall@10 为 0.9997，表示文本、图片、联合查询合计 3000 个精确 Top10 位置中，HNSW 流程找回了 2999 个。它证明索引加速几乎没有改变模型原本的排序，但不能证明模型理解得对；语义质量仍需人工标注查询或真实用户评价。

轻量回归测试：

```bash
python -m unittest discover -s tests -v
```

测试覆盖文档构建、文本冲突、共享图片、迁移一致性、双索引不同规模、候选扩召回、组内最高图片分、完整图片返回和旧格式拒绝。

## 主要文件

```text
app.py                              Gradio 页面
src/document_schema.py              文档与图片关系 schema
src/document_retrieval.py           候选展开和文档级精确重算
src/retrieval_storage.py             磁盘式 JSONL 偏移表与图文 CSR 关系
src/prepare_demo_items.py            生成 documents.jsonl
src/extract_embeddings.py            抽取文档文本和唯一图片向量
scripts/migrate_pair_artifacts.py     复用旧 pair 向量迁移到 v2
src/build_faiss_index.py              构建文本/图片 HNSW
src/search_pipeline.py                四路召回、精确重算和融合
src/hnsw_final_benchmark.py           文档级 HNSW benchmark
src/finetune/                         LoRA 训练、验证和导出
tests/                                合成数据回归测试
```

## 已知限制

- 仓库不包含可直接启动 Demo 所需的数据、模型、图片或索引；
- 当前图片相关性使用组内最高分，没有对图片顺序和跨图叙事建模；
- `src/rerank_bge.py` 是独立实验模块，尚未接入 `SearchPipeline`；
- HNSW 在 CPU 上运行，其他平台可能需要调整 PyTorch 和 FAISS 安装版本；
- 文档级人工评估尚需重新标注；
- 仓库暂未授予开源许可证，第三方模型和数据另受各自条款约束，详见 [NOTICE.md](NOTICE.md)。
