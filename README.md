# 中文多模态笔记检索

一个基于 Chinese-CLIP 和 FAISS 的中文图文检索项目。用户可以输入文字、图片，或同时输入两者；系统返回的是一篇篇完整笔记，而不是拆散的“文字—单图”数据对。

一篇笔记由一段文字和若干张图片组成：

```text
笔记
├── 文本：露营前准备的装备和食物
├── 图片 1：帐篷
├── 图片 2：炊具
└── 图片 3：食材
```

如果这篇笔记进入 Top 10，它只占一个名次，但结果中会保留三张图片。这个数据模型更接近社交媒体笔记、商品介绍和内容社区中的真实发布单元。

> 仓库只包含源码、配置和轻量测试，不包含数据集、模型权重、图片、向量或 FAISS 索引。第一次运行完整检索前，需要自行准备这些资源。
>
> 本仓库目前未授予开源许可证。公开可见不等于允许复制、修改或再分发，第三方模型和数据也受各自条款约束。详见 [NOTICE.md](NOTICE.md)。

快速导航：

- [先用合成数据跑测试](#先跑一下测试)
- [从自己的数据构建索引](#从自己的数据构建索引)
- [使用命令行或 Gradio 搜索](#开始搜索)
- [了解文档级检索原理](#检索原理)
- [区分系统指标与模型指标](#已验证结果)
- [LoRA、旧产物迁移和小内存构建](#进阶工作流)

## 这个项目能做什么

- 文本检索：输入一句中文描述，搜索相关笔记；
- 图片检索：上传一张图片，搜索文字或视觉内容相似的笔记；
- 图文联合检索：同时使用文字和图片表达查询意图；
- 笔记级排名：同一笔记被多路、多张图片命中，也只在结果中出现一次；
- 完整图片返回：排名按笔记计算，结果保留该笔记的全部图片；
- 两级检索：FAISS HNSW 快速召回，随后用原始向量精确重算候选；
- 命令行和 Gradio 页面；
- Chinese-CLIP LoRA 微调、评估和模型导出流程；
- 模型指纹、数据清单哈希和 schema 校验，防止模型、向量与索引混用。

## 与普通图文 Pair 检索的区别

假设一段文字对应三张图片。

Pair 检索会把它拆成三条记录：

```text
文字 + 图片 1
文字 + 图片 2
文字 + 图片 3
```

这样同一篇内容可能重复占据多个 Top-K 名额。本项目从数据和索引层就把它表示为一篇笔记：

```text
一个 doc_id
├── 一个文本向量
└── 多个图片向量
```

文本索引每行对应一篇笔记；图片索引每行对应一张唯一图片。图片命中后先映射到所属笔记，再以 `doc_row_id` 去重和排名，因此不是在 Top 10 生成之后才做合并。

## 先跑一下测试

轻量测试使用合成数据，不需要下载真实数据集或模型权重。这是确认环境和代码安装是否正常的最快方式。

参考环境为 Linux、Python 3.10 和 CUDA 12。请先安装与机器匹配的 PyTorch，再安装项目依赖。`requirements.txt` 默认使用 CUDA 12 版 FAISS；CPU 环境需要将它替换为兼容的 `faiss-cpu`。

```bash
git clone https://github.com/hhhhh962/multimodal_note_retrieval_zh.git
cd multimodal_note_retrieval_zh

conda create -n search python=3.10 -y
conda activate search

# 先按自己的 CUDA/CPU 环境安装 PyTorch
pip install -r requirements.txt

python -m unittest discover -s tests -v
```

当前测试共 22 项，覆盖文档构建、共享图片、重复向量校验、双索引、候选扩召回、完整图片返回和三种查询模式。

## 运行检索需要什么

完整运行需要两部分资源。

第一部分是本地 Chinese-CLIP 模型目录，例如基础模型 `OFA-Sys/chinese-clip-vit-base-patch16` 的完整 Hugging Face snapshot，或者由本项目导出的 LoRA 合并模型。目录中至少需要模型配置、处理器/分词器配置和权重文件。

第二部分是已经生成的文档级检索产物：

```text
outputs/docs_v2/
├── retrieval_meta.json
├── documents.jsonl
├── image_assets.jsonl
├── text_embeddings.npy
├── image_embeddings.npy
├── text.index
├── image.index
├── extract_state.json
├── index_meta.json
└── 若干 offset / CSR 关系数组
```

如果已经有这两部分资源，可以直接跳到“开始搜索”。否则按下一节从数据构建。

## 从自己的数据构建索引

### 1. 准备 `documents.jsonl`

检索输入是一行一篇笔记的 JSONL 文件：

```json
{
  "doc_id": "my_source:train:10001",
  "text": "露营前准备的装备和食物",
  "source": "my_source",
  "split": "train",
  "text_id": "10001",
  "images": [
    {
      "image_id": "20001",
      "image_path": "data/images/20001.jpg"
    },
    {
      "image_id": "20002",
      "image_path": "data/images/20002.jpg"
    }
  ]
}
```

约束如下：

- `doc_id` 必须唯一，建议使用 `source:split:text_id`；
- 同一笔记只有一段 `text`，可以包含多张图片；
- `image_path` 可以是绝对路径，也可以相对 `--project_root`；
- 同一路径图片只保存一份向量；
- 同一图片可以属于多篇笔记，命中后会映射到所有所属笔记；
- 同一 `doc_id` 出现文本冲突时，构建会直接报错。

仓库还提供 MUGE、Flickr30k-CN 和 COCO-CN 的数据准备脚本。它要求各数据集已经整理成 `<split>_texts.jsonl` 和 `<split>_imgs.tsv`：

```text
/path/to/datasets/
├── MUGE/
│   ├── train_texts.jsonl
│   ├── train_imgs.tsv
│   ├── valid_texts.jsonl
│   └── valid_imgs.tsv
├── Flickr30k-CN/
└── COCO-CN/
```

```bash
python src/prepare_demo_items.py \
  --datasets-root /path/to/datasets \
  --project-root . \
  --output-name demo
```

主要输出为 `data/processed/demo/documents.jsonl`。数据集和图片不会随仓库分发，请自行获取并遵守相应使用条款。

### 2. 抽取文本和图片向量

```bash
python src/extract_embeddings.py \
  --documents data/processed/demo/documents.jsonl \
  --project_root . \
  --output_dir outputs/docs_v2 \
  --model_name /path/to/chinese-clip-model \
  --device auto \
  --resume
```

脚本会生成：

- `text_embeddings.npy`：每篇笔记一行；
- `image_embeddings.npy`：每张唯一图片一行；
- 文档、图片和多对多关系元数据；
- 模型指纹、数据哈希和可续跑状态。

向量会保存为 L2 归一化的 float32。`--resume` 允许在中断后从已完成位置继续。

### 3. 构建两份 FAISS 索引

```bash
python src/build_faiss_index.py \
  --embedding_dir outputs/docs_v2 \
  --hnsw_m 32 \
  --ef_construction 200 \
  --faiss_threads 10
```

输出为：

- `text.index`：一篇笔记对应一个文本节点；
- `image.index`：一张唯一图片对应一个图片节点；
- `index_meta.json`：记录数量、维度、指纹和 HNSW 参数。

构建器会在写出索引前检查 schema、数量、维度、归一化、模型指纹和数据哈希。上面的线程数只是示例，请按机器 CPU 和内存调整。

## 开始搜索

### 命令行

文本查询：

```bash
python src/search_faiss.py \
  --embedding_dir outputs/docs_v2 \
  --project_root . \
  --model_name /path/to/chinese-clip-model \
  --query "适合户外露营的装备" \
  --top_k_final 10
```

图片查询：

```bash
python src/search_faiss.py \
  --embedding_dir outputs/docs_v2 \
  --project_root . \
  --model_name /path/to/chinese-clip-model \
  --image_path /path/to/query.jpg
```

同时传入 `--query` 和 `--image_path` 就是图文联合查询。增加 `--json` 可以输出结构化 JSON。每条结果包含笔记文本、完整 `image_paths`、图片数量、最高分匹配图片和四路分数。

### Gradio 页面

```bash
MODEL_DIR=/path/to/chinese-clip-model \
EMBEDDING_DIR=outputs/docs_v2 \
DEVICE=auto \
python app.py
```

默认监听 `0.0.0.0:7860`。结果表格一篇笔记一行，图片墙按照笔记排名展示每篇笔记的全部图片。

如果数据、缓存或产物位于其他磁盘，可以覆盖路径：

```bash
export HF_HOME=/path/to/cache/huggingface
export HF_DATASETS_CACHE=/path/to/cache/huggingface/datasets
export PIP_CACHE_DIR=/path/to/cache/pip
export DOCUMENTS=/path/to/documents.jsonl
export OUTPUT_DIR=/path/to/outputs/docs_v2
```

## 检索原理

```text
文本查询 ─┬─> 文本 HNSW：直接得到候选笔记 ─────┐
          └─> 图片 HNSW：图片命中映射到候选笔记 ─┤
                                                 ├─> 唯一候选笔记集合
图片查询 ─┬─> 文本 HNSW：直接得到候选笔记 ─────┤
          └─> 图片 HNSW：图片命中映射到候选笔记 ─┘
                                                          │
                                                          v
                                 使用候选笔记的文本和全部图片精确重算
                                                          │
                                                          v
                                             分数融合后返回唯一笔记 Top-K
```

候选笔记会计算四路相似度：

- `text → text`：查询文本与笔记文本；
- `text → image`：查询文本与笔记所有图片的最高分；
- `image → text`：查询图片与笔记文本；
- `image → image`：查询图片与笔记所有图片的最高分。

图片路线使用组内最高分，避免多图笔记因为包含辅助图而被平均分稀释。`alpha` 控制文本索引分与图片索引分的融合比例，默认值为 `0.5`。

图片索引会从 `4 × top_k_recall` 张图片开始召回。如果映射后仍没有足够的唯一笔记，召回数量会自动翻倍，直到达到目标或索引耗尽。

默认索引为内积 `IndexHNSWFlat`，参数为 `M=32`、`efConstruction=200`、`efSearch=512`。HNSW 保持在 CPU；GPU 用于 Chinese-CLIP 查询编码。

## 已验证结果

下面的数据是项目全量服务器运行的固定快照。生成这些结果的模型、向量和索引没有包含在 GitHub 仓库中。

### 文档级产物

| 项目 | 数量 |
| --- | ---: |
| 笔记 | 434,927 |
| 唯一图片 | 211,310 |
| 笔记—图片关系 | 462,035 |
| 向量维度 | 512 |

文本、图片和图文联合三种真实查询均已完成启动与 Top 10 集成验证，22 项轻量测试全部通过。

### HNSW 是否改变了精确排名

这个 benchmark 只检查 **FAISS HNSW 加速有没有漏掉精确结果**，不评价结果对用户是否相关。

测试先让查询与全库 434,927 篇笔记逐一计算，得到精确 Top-K；再运行实际的 HNSW 召回和文档级重排，比较两份 Top-K 有多少篇相同。

```bash
python src/hnsw_final_benchmark.py \
  --embedding_dir outputs/docs_v2 \
  --sample_size 100 \
  --candidate_pool_size 100
```

固定 100 篇样本、候选池 100、`efSearch=512` 的结果：

| 查询方式 | Recall@5 | Recall@10 |
| --- | ---: | ---: |
| 文本 | 1.000 | 1.000 |
| 图片 | 1.000 | 1.000 |
| 文本 + 图片 | 1.000 | 0.999 |
| 三种方式平均 | 1.000 | 0.9997 |

总体 Recall@10 为 `0.9997`：三种查询合计 3000 个精确 Top 10 位置中，HNSW 流程找回了 2999 个。它说明索引加速几乎没有改变模型原本的排序，不能说明模型理解得是否正确。

### LoRA 模型指标

以下指标评价 Chinese-CLIP 模型在五个数据集检索任务上的表现，不是文档级系统人工评价。

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

文档级系统目前没有可公开复现的人工相关性指标。要判断结果对真实用户是否有用，仍需重新标注查询或收集线上反馈。

## 进阶工作流

下面三部分面向需要训练模型、迁移旧系统或在极小内存机器上建图的读者。只使用现成模型和索引时可以跳过。

<details>
<summary><strong>LoRA 微调</strong></summary>

### LoRA 微调

训练使用多正样本损失，一段文本对应多张图片时不需要拆成单正样本训练。完整说明见 [FINETUNE.md](FINETUNE.md)。

```bash
bash scripts/run_finetune_lora.sh
bash scripts/show_finetune_progress.sh
bash scripts/run_finetune_eval.sh valid
bash scripts/run_finetune_export.sh
bash scripts/run_rebuild_finetuned.sh
```

默认配置为 `configs/finetune_lora.yaml`。可以通过 `PYTHON`、`CONFIG`、`MODEL_DIR`、`DOCUMENTS` 和 `FINETUNED_INDEX_DIR` 覆盖脚本路径。

</details>

<details>
<summary><strong>从旧 Pair 产物迁移</strong></summary>

### 从旧 Pair 产物迁移

只有已经持有旧版一一对齐的文本/图片向量时才需要这一节。迁移工具可以复用向量，不重新占用 GPU 编码：

```bash
python scripts/migrate_pair_artifacts.py \
  --source-dir /path/to/old_pair_artifacts \
  --items data/processed/demo/items.jsonl \
  --output-dir outputs/docs_v2

python src/build_faiss_index.py \
  --embedding_dir outputs/docs_v2
```

迁移会验证同一笔记的重复文本向量和同一路径的重复图片向量是否一致，并拒绝覆盖非空目标目录。默认容差为 `atol=1e-6`；只有确认旧 FP16 分批编码存在微小漂移后，才应显式提高容差。

</details>

<details>
<summary><strong>约 2 GB 内存下构建索引</strong></summary>

### 小内存构建

标准索引构建需要同时保存向量和 HNSW 图。如果机器内存约为 2 GB，可以使用：

```bash
python src/build_faiss_index.py \
  --embedding_dir outputs/docs_v2 \
  --low_memory_build \
  --faiss_threads 1 \
  --chunk_size 5000
```

该模式只在建图阶段使用 FP16 scalar-quantized 存储，最终仍写出标准 float32 `IndexHNSWFlat`。如果文本图已经转存到磁盘、但最终装配被中断，可以增加 `--resume_text_spooled_graph` 恢复。

</details>

## 主要文件

```text
app.py                              Gradio 页面
configs/search.yaml                 检索默认配置
configs/finetune_lora.yaml          LoRA 训练配置
src/document_schema.py              笔记与图片关系 schema
src/retrieval_storage.py            JSONL 偏移表和图文 CSR 关系
src/extract_embeddings.py           抽取笔记文本与唯一图片向量
src/build_faiss_index.py             构建文本/图片 HNSW
src/document_retrieval.py           候选映射和文档级精确重算
src/search_pipeline.py               检索主流程
src/search_faiss.py                  命令行入口
src/hnsw_final_benchmark.py          HNSW 与全库精确排名对比
src/finetune/                        LoRA 训练、评估和导出
scripts/migrate_pair_artifacts.py     旧 Pair 向量迁移
tests/                                合成数据回归测试
```

## 已知限制

- 仓库不能开箱即用地展示真实搜索结果，因为数据、模型和索引不在 Git 中；
- 当前用组内最高图片相似度代表一篇多图笔记，没有建模图片顺序和跨图叙事；
- HNSW benchmark 只衡量近似索引相对精确计算的损失，不衡量语义相关性；
- 当前文档级人工评价尚未重新标注；
- `src/rerank_bge.py` 是独立实验模块，尚未接入主检索流程；
- 默认依赖面向 Linux 和 CUDA 12，其他平台需要调整 PyTorch 与 FAISS；
- 仓库暂未授予开源许可证，第三方模型和数据另受各自条款约束。

第三方资源和权利边界见 [NOTICE.md](NOTICE.md)。
