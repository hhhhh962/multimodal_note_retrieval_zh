# Project 1: 中文多模态笔记搜索系统

本项目构建一个中文多模态搜索项目：

```text
中文 query
  -> Chinese-CLIP 提取 query 向量
  -> FAISS GPU 召回相关图文笔记
  -> BGE reranker 重排候选结果
  -> 返回 top10 图文笔记
  -> 使用 Recall@K、MRR、NDCG 评估搜索效果
```

## 1. Environment

参考环境：

```text
conda env: search
python: 3.10
torch: 2.8.0+cu128
torchvision: 0.23.0+cu128
faiss-gpu-cu12: 1.14.1.post1
numpy: 2.2.6
opencv-python-headless: 4.13.0.92
transformers: 5.12.1
FlagEmbedding: 1.4.0
gradio: 6.19.0
```

创建并激活环境：

```bash
conda create -n search python=3.10 -y
conda activate search
pip install -r requirements.txt
```

如需把缓存放到自定义磁盘，可在运行前设置：

```bash
export CACHE_ROOT="/path/to/cache"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
```

## 2. Project Structure

```text
project_1_multimodal_search/
  README.md
  requirements.txt
  app.py

  data/
    raw/
    images/
    processed/

  src/
    prepare_data.py
    extract_embeddings.py
    build_faiss_index.py
    search_faiss.py
    rerank_bge.py
    search_pipeline.py
    evaluate.py
    utils.py

  outputs/

  reports/
    experiment_report.md
    bad_cases.md

  assets/

  configs/
    search.yaml
```

## 3. Data Format

第一步直接准备真实图文数据。

核心文件：

```text
data/processed/notes.jsonl
```

每行一条图文笔记：

```json
{"note_id":"000001","title":"夏日通勤穿搭","content":"白衬衫搭配浅色牛仔裤，适合夏天上班。","image_path":"data/images/000001.jpg","tags":["穿搭","通勤","夏天"]}
```

字段说明：

```text
note_id：笔记唯一 ID
title：标题
content：正文
image_path：图片路径
tags：标签列表
```

评估文件：

```text
data/processed/queries.jsonl
```

每行一个 query：

```json
{"query":"夏天上班穿什么","positive_note_ids":["000001","000018"]}
```

## 4. Implementation Roadmap

### Step 1: 准备真实图文数据

先准备 100-1000 条真实图文样本，不做 toy 数据。

要求：

```text
图片放在 data/images/
标准笔记写入 data/processed/notes.jsonl
每条样本必须有 note_id、title、content、image_path
随机抽查图片路径是否能打开
```

### Step 2: 提取 Chinese-CLIP 图文向量

实现文件：

```text
src/extract_embeddings.py
```

目标输出：

```text
outputs/text_embeddings.npy
outputs/image_embeddings.npy
outputs/note_embeddings.npy
outputs/note_meta.json
```

第一版融合方式：

```text
note_embedding = 0.5 * text_embedding + 0.5 * image_embedding
```

### Step 3: 构建 FAISS HNSW 索引

实现文件：

```text
src/build_faiss_index.py
```

目标输出：

```text
outputs/finetuned_full/text.index
outputs/finetuned_full/image.index
```

第一版使用：

```text
IndexHNSWFlat(M=32, efConstruction=200) + normalized embeddings
```

HNSW 图索引运行在 CPU；Chinese-CLIP 仍在 GPU 上编码每次新查询。正式参数固定为
`M=32`、`efConstruction=200`、`efSearch=512`，不再遍历其他查询参数。评估时从现有
embedding 在内存中临时构造 `IndexFlatIP` 真值，但不会保存 Flat 文件。

质量报告只统计每路 Top-100 候选经过线上融合和图片去重后的最终 Recall@5/Recall@10，
覆盖文字、图片、图文联合三种查询模式。可在不重建索引的情况下重新生成报告：

```bash
python -u src/hnsw_final_benchmark.py \
  --embedding_dir outputs/finetuned_full \
  --sample_size 1000 \
  --candidate_pool_size 100
```

本次固定参数评估结果：总体 Recall@5 为 `0.9874`，Recall@10 为 `0.9882`。
30 条查询经逐图复核后的指标为 P@5 `0.9733`、P@10 `0.9333`、
NDCG@5 `0.9567`、NDCG@10 `0.9456`。

替换已有 Flat 索引：

```bash
python -u src/build_faiss_index.py \
  --embedding_dir outputs/finetuned_full \
  --output_dir outputs/finetuned_full \
  --hnsw_m 32 \
  --ef_construction 200 \
  --replace_existing
```

### Step 4: 实现 FAISS 搜索

实现文件：

```text
src/search_faiss.py
```

目标：

```text
输入中文 query
提取 query embedding
从 FAISS 中召回 topK 图文笔记
返回 note_id、title、image_path、score
```

### Step 5: 接入 BGE reranker

实现文件：

```text
src/rerank_bge.py
```

流程：

```text
FAISS 召回 top50
  -> BGE reranker 重排
  -> 输出 top10
```

### Step 6: 串联完整搜索链路

实现文件：

```text
src/search_pipeline.py
```

完整流程：

```text
query
  -> Chinese-CLIP query embedding
  -> FAISS GPU recall
  -> BGE reranker
  -> final top10
```

### Step 7: 评估指标

实现文件：

```text
src/evaluate.py
```

评估指标：

```text
Recall@5
Recall@10
MRR
NDCG@10
平均检索耗时
```

### Step 8: Gradio Demo

实现文件：

```text
app.py
```

页面功能：

```text
输入中文 query
展示 top10 图文笔记
展示图片、标题、正文摘要、标签、分数
```

## 5. Reports

`reports/experiment_report.md` 记录：

```text
数据集来源
样本规模
模型选择
图文融合方式
搜索指标
耗时表现
最终结论
```

`reports/bad_cases.md` 记录：

```text
query
错误结果
理想结果
失败原因
改进思路
```

## 6. Current First Target

当前第一阶段目标：

```text
用真实 notes.jsonl 跑通 Chinese-CLIP + FAISS GPU 的端到端搜索闭环。
```

验收标准：

```text
1. data/processed/notes.jsonl 至少有 100 条真实图文笔记
2. outputs/note_embeddings.npy 成功生成
3. outputs/notes.index 成功生成
4. 输入中文 query 可以返回 top10 图文笔记
5. 返回结果包含 note_id、title、image_path、score
```

完成后再接 BGE reranker、评估指标和 Gradio Demo。

## LoRA 微调入口

三数据集 Chinese-CLIP 双塔 LoRA 微调的配置、启动方式、进度查看、断点恢复、评估、模型导出和索引重建说明见 [FINETUNE.md](FINETUNE.md)。真正启动训练使用 `bash scripts/run_finetune_lora.sh`；训练不会自动执行后处理。当前默认检索版本为 `outputs/finetune_lora/exported_model` 与 `outputs/finetuned_full`。
