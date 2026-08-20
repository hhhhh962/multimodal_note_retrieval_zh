# Chinese-CLIP LoRA 微调运行说明

这套流程在 MUGE、Flickr30k-CN、COCO-CN 三套官方 LMDB 上微调 Chinese-CLIP ViT-B/16 的双塔注意力层，同时训练两个投影层和 `logit_scale`。训练、评估、导出和全量索引重建彼此分开；启动训练后不会自动执行后续步骤。

## 1. “远程测试”和“真正微调”的区别

远程测试是把代码放到远程项目环境后，检查依赖、路径、数据读取、模型加载、CLI 和小规模前后向是否能工作。它的目标是尽早发现环境或接口问题，不代表已经开始完整训练，也不会产生可交付的最终模型。

真正微调会读取三套完整训练 LMDB，在 GPU 上持续执行两个 epoch，定期跑三数据集全量 valid 检索，保存 checkpoint 并选择最佳模型。这一步耗时长、占用 GPU，只有明确执行 `run_finetune_lora.sh` 才会开始。

## 2. BF16、FP32 与混合精度

`training.precision: auto` 使用的是混合精度策略：模型的关键参数和优化器状态保持适合训练的精度，而矩阵乘等计算在 `autocast` 中优先使用 BF16。BF16 是一种 16 位数值格式；“混合精度”是哪些操作用 16 位、哪些保留 32 位的运行策略，两者不是互斥选项。

支持 BF16 的 CUDA 显卡会自动选 BF16；不支持时退回 FP16，并启用 `GradScaler`；CPU 自动模式使用 FP32。BF16 与 FP32 有相同数量级的指数范围，通常比 FP16 更不容易上溢或下溢，同时显著降低激活显存和计算开销。全程 FP32 更稳但显存占用更高、速度更慢，会压低可用 micro-batch。若排查数值问题，可以临时把配置改成 `precision: fp32`，正式方案仍建议保留 `auto`。

## 3. 路径配置与输出隔离

默认配置文件是 `configs/finetune_lora.yaml`，路径相对于仓库根目录：

```text
项目：当前仓库根目录
数据：data/{MUGE,Flickr30k-CN,COCO-CN}
基座：OFA-Sys/chinese-clip-vit-base-patch16
训练输出：outputs/finetune_lora
当前正式模型：outputs/finetune_lora/exported_model
当前正式全量索引：outputs/finetuned_full
```

## 4. 开始微调

进入项目并执行：

```bash
cd /path/to/project_1_multimodal_search
bash scripts/run_finetune_lora.sh
```

脚本用 `nohup` 后台启动，SSH 断开后任务仍会继续，并写出：

```text
outputs/finetune_lora/train.log       完整标准输出和错误日志
outputs/finetune_lora/train.pid       后台进程号
outputs/finetune_lora/exit_code       结束后生成；0 成功，非 0 失败
outputs/finetune_lora/progress.json   当前进度快照
outputs/finetune_lora/metrics.jsonl   逐事件历史记录
outputs/finetune_lora/checkpoints/    定期 checkpoint
outputs/finetune_lora/best_checkpoint.json  最佳 checkpoint 指针
```

## 5. 查看微调进度

单次查看最推荐用：

```bash
bash scripts/show_finetune_progress.sh
```

每 10 秒自动刷新：

```bash
watch -n 10 bash scripts/show_finetune_progress.sh
```

只持续跟踪原始日志：

```bash
tail -f outputs/finetune_lora/train.log
```

进度脚本会同时展示进程/退出码、epoch、优化步百分比、loss、实际精度、micro-batch、梯度累计、吞吐、ETA、验证分数、最佳 checkpoint、训练进程显存统计、最新日志和当前 `nvidia-smi`。

看到以下任一状态即可判断任务已经结束：

```text
completed      正常跑完
early_stopped  触发早停，属于正常结束
failed         失败，需要先看失败原因和 train.log
```

真正开始微调后，应把上面的进度命令告诉用户，然后停止后续工作，等待用户确认远程训练完成。不要自行运行本说明第 7～9 节的评估、导出和重建脚本。

## 6. 中断后恢复

恢复输出目录中最新 checkpoint：

```bash
bash scripts/run_finetune_lora.sh --resume
```

恢复指定 checkpoint：

```bash
bash scripts/run_finetune_lora.sh --resume outputs/finetune_lora/checkpoints/step-00000250
```

恢复时 micro-batch、梯度累计、有效 batch 和总训练步数必须与原任务一致。不要同时启动两个写入同一输出目录的训练进程。

## 7. 训练完成后评估（等待用户让你继续）

评估最佳 checkpoint 的 valid：

```bash
bash scripts/run_finetune_eval.sh valid
```

评估 test，并为没有答案的 MUGE test 生成提交 JSONL：

```bash
bash scripts/run_finetune_eval.sh test
```

也可以把 checkpoint 目录作为第二个参数传入。报告写在 `outputs/finetune_lora/evaluation/`。

## 8. 导出最佳模型（评估确认后再做）

```bash
bash scripts/run_finetune_export.sh
```

默认把最佳 adapter 与基座合并到 `outputs/finetune_lora/exported_model`，随后从磁盘按普通 `ChineseCLIPModel` 重载，并检查文本塔和图片塔都能输出有限的 512 维向量。导出目录非空时脚本会拒绝覆盖。

## 9. 用微调模型重建全量向量和索引（导出确认后再做）

```bash
bash scripts/run_rebuild_finetuned.sh
tail -f outputs/finetune_lora/rebuild.log
```

新向量和索引只写入 `outputs/finetuned_full`。任务结束后查看：

```bash
cat outputs/finetune_lora/rebuild_exit_code
```

输出为 `0` 表示向量抽取和两个 FAISS 索引均成功。当前正式检索入口默认加载 `outputs/finetune_lora/exported_model` 和 `outputs/finetuned_full`；原始基线索引已删除，如需回退必须使用保留的基座缓存重新抽取向量并构建索引。
