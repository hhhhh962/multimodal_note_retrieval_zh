"""Chinese-CLIP 双塔 LoRA 微调训练入口。

训练配置来自 YAML。入口负责三套 LMDB 的温度平衡采样、轻量图像增强、
自动显存探测、混合精度、梯度累计、全量检索验证、早停及完整断点恢复。
对比损失只在当前 micro-batch 内构造负样本；梯度累计不会把不同
micro-batch 的特征拼成一个更大的负样本池。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.finetune"

from .checkpointing import (
    METADATA_FILENAME,
    ProgressReporter,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from .data import (
    LMDBPairDataset,
    MultiPositiveCollator,
    MultiSourceDataset,
    TemperatureBalancedSampler,
)
from .evaluate_retrieval import evaluate_checkpoint
from .losses import symmetric_multi_positive_loss
from .modeling import (
    DEFAULT_MODEL_NAME,
    apply_dual_tower_lora,
    clamp_logit_scale_,
    enable_gradient_checkpointing,
    encode_training_batch,
    get_base_chinese_clip,
    load_chinese_clip_model,
    parameter_report,
    split_trainable_parameter_groups,
    trainable_gradients_are_finite,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_MICRO_BATCH_CANDIDATES = (64, 32, 16)


@dataclass(frozen=True)
class PrecisionPolicy:
    """一次训练运行实际采用的数值精度策略。"""

    name: str
    autocast_dtype: torch.dtype | None
    use_grad_scaler: bool


@dataclass(frozen=True)
class MicroBatchProbe:
    """单个自动 micro-batch 候选的显存探测结果。"""

    batch_size: int
    accepted: bool
    peak_memory_bytes: int | None
    memory_limit_bytes: int | None
    reason: str | None = None


def load_yaml_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """读取 YAML/JSON 配置，并记录相对路径解析所需的配置目录。"""

    if isinstance(config, Mapping):
        loaded = dict(config)
        loaded.setdefault("_config_dir", str(Path.cwd()))
        return loaded

    path = Path(config).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取训练 YAML 需要安装 pyyaml") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置根节点必须是对象：{path}")
    result = dict(loaded)
    result["_config_dir"] = str(path.parent)
    result["_config_path"] = str(path)
    return result


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """读取一个可选配置段，并拒绝意外的非对象值。"""

    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"配置段 {name!r} 必须是对象")
    return dict(value)


def _resolve_path(value: str | Path, config: Mapping[str, Any]) -> Path:
    """让训练输入输出中的相对路径统一相对 YAML 所在目录解析。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(str(config.get("_config_dir", Path.cwd()))) / path).resolve()


def _resolve_model_name(value: str | Path, config: Mapping[str, Any]) -> str:
    """本地模型路径相对 YAML 解析，Hub 模型名保持原样。"""

    text = str(value)
    candidate = _resolve_path(text, config)
    if candidate.exists():
        return str(candidate)
    return text


def seed_everything(seed: int) -> None:
    """初始化 Python、NumPy、CPU 与全部 CUDA 设备的随机状态。"""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_precision(
    requested: str,
    device: str | torch.device,
    *,
    bf16_supported: bool | None = None,
) -> PrecisionPolicy:
    """CUDA 自动模式优先 BF16，不支持时才使用带 GradScaler 的 FP16。"""

    run_device = torch.device(device)
    value = str(requested).strip().lower()
    aliases = {
        "float32": "fp32",
        "none": "fp32",
        "float16": "fp16",
        "bfloat16": "bf16",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"未知 training.precision={requested!r}")

    if run_device.type != "cuda":
        if value in {"auto", "fp32"}:
            return PrecisionPolicy("fp32", None, False)
        if value == "bf16" and run_device.type == "cpu":
            return PrecisionPolicy("bf16", torch.bfloat16, False)
        raise RuntimeError(f"{run_device.type} 设备不支持本入口的 {value} 混合精度")

    if bf16_supported is None:
        bf16_supported = bool(torch.cuda.is_bf16_supported())
    if value == "auto":
        value = "bf16" if bf16_supported else "fp16"
    if value == "bf16":
        if not bf16_supported:
            raise RuntimeError("请求 BF16，但当前 CUDA 设备不支持 BF16")
        return PrecisionPolicy("bf16", torch.bfloat16, False)
    if value == "fp16":
        return PrecisionPolicy("fp16", torch.float16, True)
    return PrecisionPolicy("fp32", None, False)


def _autocast_context(device: torch.device, policy: PrecisionPolicy):
    """按已解析的策略创建 autocast 上下文。"""

    if policy.autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=policy.autocast_dtype)


def _make_grad_scaler(policy: PrecisionPolicy) -> Any | None:
    """仅为 CUDA FP16 创建 GradScaler，兼容新旧 PyTorch 接口。"""

    if not policy.use_grad_scaler:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def compute_accumulation_steps(effective_batch_size: int, micro_batch_size: int) -> int:
    """计算普通梯度累计次数，并保证名义有效 batch 精确可达。"""

    effective = int(effective_batch_size)
    micro = int(micro_batch_size)
    if effective <= 0 or micro <= 0:
        raise ValueError("effective_batch_size 和 micro_batch_size 必须为正整数")
    if effective < micro or effective % micro:
        raise ValueError(
            "effective_batch_size 必须是不小于 micro_batch_size 的整数倍；"
            f"当前为 {effective} 和 {micro}"
        )
    return effective // micro


def accumulation_group_sizes(num_micro_batches: int, accumulation_steps: int) -> tuple[int, ...]:
    """返回一个 epoch 的累计组大小，最后一组使用真实剩余批次数。"""

    total = int(num_micro_batches)
    width = int(accumulation_steps)
    if total < 0 or width <= 0:
        raise ValueError("num_micro_batches 不能为负，accumulation_steps 必须为正")
    full, remainder = divmod(total, width)
    groups = [width] * full
    if remainder:
        groups.append(remainder)
    return tuple(groups)


def cosine_warmup_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """计算线性 warmup 后余弦衰减的统一学习率倍率。"""

    current = max(int(step), 0)
    total = max(int(total_steps), 1)
    warmup = min(max(int(warmup_steps), 0), total)
    if warmup and current < warmup:
        return float(current) / float(max(warmup, 1))
    progress = float(current - warmup) / float(max(total - warmup, 1))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.05,
) -> tuple[torch.optim.lr_scheduler.LambdaLR, int]:
    """创建所有 AdamW 参数组共享倍率的 5% warmup 余弦计划。"""

    ratio = float(warmup_ratio)
    if not 0.0 <= ratio < 1.0:
        raise ValueError("optimizer.warmup_ratio 必须位于 [0, 1)")
    total = int(total_steps)
    if total <= 0:
        raise ValueError("total_steps 必须为正整数")
    warmup = int(total * ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(step, total, warmup),
    )
    return scheduler, warmup


def _is_cuda_oom(error: BaseException) -> bool:
    """兼容不同 PyTorch 版本的 CUDA OOM 异常类型和消息。"""

    out_of_memory = getattr(torch.cuda, "OutOfMemoryError", ())
    if out_of_memory and isinstance(error, out_of_memory):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def select_micro_batch_size(
    configured: int | str,
    candidates: Sequence[int],
    *,
    memory_fraction: float,
    probe: Callable[[int], int | None],
    total_memory_bytes: int | None,
) -> tuple[int, tuple[MicroBatchProbe, ...]]:
    """按给定顺序实测候选，OOM 或峰值超过显存阈值时尝试下一档。"""

    fraction = float(memory_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("training.memory_fraction 必须位于 (0, 1]")
    if not (isinstance(configured, str) and configured.strip().lower() == "auto"):
        selected = int(configured)
        if selected <= 0:
            raise ValueError("training.micro_batch_size 必须为正整数或 auto")
        return selected, ()

    ordered = tuple(int(value) for value in candidates)
    if not ordered or any(value <= 0 for value in ordered):
        raise ValueError("training.micro_batch_candidates 必须包含正整数")
    limit = (
        int(int(total_memory_bytes) * fraction)
        if total_memory_bytes is not None
        else None
    )
    attempts: list[MicroBatchProbe] = []
    for candidate in ordered:
        try:
            peak = probe(candidate)
        except BaseException as exc:
            if not _is_cuda_oom(exc):
                raise
            attempts.append(
                MicroBatchProbe(candidate, False, None, limit, "cuda_oom")
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        accepted = limit is None or peak is None or int(peak) <= limit
        attempts.append(
            MicroBatchProbe(
                candidate,
                accepted,
                None if peak is None else int(peak),
                limit,
                None if accepted else "memory_threshold_exceeded",
            )
        )
        if accepted:
            return candidate, tuple(attempts)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    detail = ", ".join(f"{item.batch_size}:{item.reason}" for item in attempts)
    raise RuntimeError(f"自动 micro-batch 探测全部失败（{detail}）")


def build_train_image_transform(data_config: Mapping[str, Any]):
    """创建 PIL 级 RandomResizedCrop 与随机水平翻转增强。"""

    try:
        from torchvision.transforms import (
            Compose,
            InterpolationMode,
            RandomHorizontalFlip,
            RandomResizedCrop,
        )
    except ImportError as exc:
        raise RuntimeError("训练图像增强需要安装 torchvision") from exc

    image_size = int(data_config.get("image_size", 224))
    scale_value = data_config.get("train_crop_scale", (0.8, 1.0))
    if not isinstance(scale_value, Sequence) or isinstance(scale_value, (str, bytes)):
        raise ValueError("data.train_crop_scale 必须是两个浮点数")
    scale = tuple(float(value) for value in scale_value)
    if len(scale) != 2 or not 0.0 < scale[0] <= scale[1] <= 1.0:
        raise ValueError("data.train_crop_scale 必须满足 0 < min <= max <= 1")
    probability = float(data_config.get("hflip_probability", 0.5))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("data.hflip_probability 必须位于 [0, 1]")
    if image_size <= 0:
        raise ValueError("data.image_size 必须为正整数")
    return Compose(
        [
            RandomResizedCrop(
                image_size,
                scale=scale,
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomHorizontalFlip(p=probability),
        ]
    )


def _training_device(training_config: Mapping[str, Any]) -> torch.device:
    """解析训练设备；auto 在有 CUDA 时使用当前卡，否则回退 CPU。"""

    requested = str(training_config.get("device", "auto"))
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但当前进程看不到可用 CUDA 设备")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _configure_cuda_limit(device: torch.device, memory_fraction: float) -> int | None:
    """在加载模型前设置进程显存上限，并返回物理显存总字节数。"""

    if device.type != "cuda":
        return None
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    try:
        torch.cuda.set_per_process_memory_fraction(float(memory_fraction), device=device)
    except (AttributeError, RuntimeError) as exc:
        LOGGER.warning("无法设置 CUDA 进程显存比例上限：%s", exc)
    return int(properties.total_memory)


def _train_source_specs(
    data_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[tuple[str, Path], ...]:
    """读取并严格校验三套训练 LMDB 的来源名和路径。"""

    values = data_config.get("train_sources")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("data.train_sources 必须是三个对象组成的列表")
    specs: list[tuple[str, Path]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise ValueError(f"data.train_sources[{index}] 必须是对象")
        source = str(value.get("source") or value.get("name") or "").strip()
        raw_path = value.get("lmdb_path") or value.get("train_lmdb") or value.get("path")
        if not source or raw_path is None:
            raise ValueError(
                f"data.train_sources[{index}] 必须同时提供 source 和 lmdb_path"
            )
        specs.append((source, _resolve_path(str(raw_path), config)))
    if len(specs) != 3:
        raise ValueError(f"训练必须配置三套 LMDB，当前数量为 {len(specs)}")
    if len({source for source, _ in specs}) != 3:
        raise ValueError("三套训练 LMDB 的 source 必须互不相同")
    return tuple(specs)


def _build_training_dataset(
    data_config: Mapping[str, Any],
    config: Mapping[str, Any],
) -> MultiSourceDataset:
    """构造三套增强后的 LMDB 数据集并合并完整多正样本关系。"""

    transform = build_train_image_transform(data_config)
    datasets = [
        LMDBPairDataset(path, source=source, image_transform=transform)
        for source, path in _train_source_specs(data_config, config)
    ]
    return MultiSourceDataset(datasets)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """只把张量字段搬到训练设备，UID 与原始元数据留在主存。"""

    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _probe_callable(
    dataset: MultiSourceDataset,
    collator: MultiPositiveCollator,
    model: torch.nn.Module,
    device: torch.device,
    policy: PrecisionPolicy,
) -> Callable[[int], int | None]:
    """创建不推进 sampler、不保留梯度且恢复 RNG 的真实前后向探测函数。"""

    def probe(batch_size: int) -> int | None:
        rng_state = capture_rng_state()
        was_training = model.training
        model.train()
        model.zero_grad(set_to_none=True)
        temporary_scaler = _make_grad_scaler(policy)
        try:
            samples = [dataset[index % len(dataset)] for index in range(batch_size)]
            batch = _move_batch(collator(samples), device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            with _autocast_context(device, policy):
                text_features, image_features = encode_training_batch(model, batch)
                loss = symmetric_multi_positive_loss(
                    text_features,
                    image_features,
                    batch["positive_mask"],
                    get_base_chinese_clip(model).logit_scale,
                )
            if temporary_scaler is not None:
                temporary_scaler.scale(loss).backward()
            else:
                loss.backward()
            if device.type != "cuda":
                return None
            torch.cuda.synchronize(device)
            return max(
                int(torch.cuda.max_memory_allocated(device)),
                int(torch.cuda.max_memory_reserved(device)),
            )
        finally:
            model.zero_grad(set_to_none=True)
            if not was_training:
                model.eval()
            del temporary_scaler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            restore_rng_state(rng_state)

    return probe


def _seed_worker(worker_id: int) -> None:
    """让 DataLoader 子进程的 NumPy 与 Python 随机增强获得独立种子。"""

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _make_loader(
    dataset: MultiSourceDataset,
    sampler: TemperatureBalancedSampler,
    collator: MultiPositiveCollator,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """按当前 sampler 游标创建一轮 DataLoader，worker 预取不修改恢复游标。"""

    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(sampler.epoch) * 1_000_003)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "drop_last": False,
        "num_workers": int(num_workers),
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
        "generator": generator,
        "persistent_workers": False,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _gpu_metrics(device: torch.device, total_memory: int | None) -> dict[str, Any]:
    """采集当前显存、保留显存、峰值和相对物理显存比例。"""

    if device.type != "cuda":
        return {
            "gpu_memory_allocated_bytes": 0,
            "gpu_memory_reserved_bytes": 0,
            "gpu_memory_peak_bytes": 0,
            "gpu_memory_total_bytes": 0,
            "gpu_memory_fraction": 0.0,
        }
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    peak = max(
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )
    return {
        "gpu_memory_allocated_bytes": allocated,
        "gpu_memory_reserved_bytes": reserved,
        "gpu_memory_peak_bytes": peak,
        "gpu_memory_total_bytes": int(total_memory) if total_memory else None,
        "gpu_memory_fraction": float(peak / total_memory) if total_memory else None,
    }


def _learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """按三组稳定名称导出当前学习率。"""

    return {
        str(group.get("group_name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def _checkpoint_dirs(output_dir: Path) -> list[Path]:
    """列出所有具有完整 metadata 的可恢复 checkpoint。"""

    root = output_dir / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and (path / METADATA_FILENAME).is_file()
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def resolve_resume_checkpoint(
    value: str | Path | bool | None,
    output_dir: Path,
    config: Mapping[str, Any],
) -> Path | None:
    """把 false/路径/latest 统一解析为一个完整 checkpoint 目录。"""

    if value is None or value is False or str(value).strip().lower() in {"", "false", "none"}:
        return None
    if value is True or str(value).strip().lower() in {"true", "latest", "auto"}:
        candidates = _checkpoint_dirs(output_dir)
        if not candidates:
            raise FileNotFoundError(f"{output_dir} 中没有可恢复 checkpoint")
        return candidates[-1]
    path = _resolve_path(str(value), config)
    if not (path / METADATA_FILENAME).is_file():
        raise FileNotFoundError(f"checkpoint 不完整或不存在：{path}")
    return path


def _metadata_trainer_state(checkpoint: Path | None) -> dict[str, Any]:
    """在建优化器前读取 checkpoint 的轻量状态，用于固定 micro-batch。"""

    if checkpoint is None:
        return {}
    with (checkpoint / METADATA_FILENAME).open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    value = metadata.get("trainer_state", {})
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint metadata 的 trainer_state 非法：{checkpoint}")
    return dict(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """以替换方式写入 best checkpoint 指针，避免监控端读取半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _close_dataset(dataset: MultiSourceDataset | None) -> None:
    """显式关闭主进程可能打开的三套 LMDB 只读句柄。"""

    if dataset is None:
        return
    for child in dataset.datasets:
        child.close()


def train(
    config: str | Path | Mapping[str, Any],
    *,
    resume: str | Path | bool | None = None,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_checkpoint,
) -> dict[str, Any]:
    """执行一次可恢复训练，并返回与 progress.json 一致的最终摘要。"""

    loaded = load_yaml_config(config)
    model_config = _section(loaded, "model")
    data_config = _section(loaded, "data")
    training_config = _section(loaded, "training")
    optimizer_config = _section(loaded, "optimizer")
    lora_config = _section(model_config, "lora")

    output_dir = _resolve_path(
        str(training_config.get("output_dir", "outputs/finetune")), loaded
    )
    reporter = ProgressReporter(output_dir)
    dataset: MultiSourceDataset | None = None
    try:
        seed = int(training_config.get("seed", 20_260_731))
        epochs = int(training_config.get("epochs", 2))
        epoch_size = int(data_config.get("epoch_size", 419_294))
        num_workers = int(data_config.get("num_workers", 8))
        effective_batch_size = int(training_config.get("effective_batch_size", 256))
        memory_fraction = float(training_config.get("memory_fraction", 0.90))
        eval_every = int(training_config.get("eval_every_steps", 250))
        patience = int(training_config.get("early_stopping_patience", 4))
        gradient_clip = float(training_config.get("gradient_clip", 1.0))
        if epochs <= 0 or epoch_size <= 0 or num_workers < 0:
            raise ValueError("epochs/epoch_size 必须为正，num_workers 不能为负")
        if eval_every <= 0 or patience <= 0 or gradient_clip <= 0:
            raise ValueError("eval_every_steps、early_stopping_patience、gradient_clip 必须为正")

        seed_everything(seed)
        device = _training_device(training_config)
        total_memory = _configure_cuda_limit(device, memory_fraction)
        policy = resolve_training_precision(
            str(training_config.get("precision", "auto")), device
        )
        resume_value = resume if resume is not None else training_config.get("resume")
        resume_checkpoint = resolve_resume_checkpoint(
            resume_value, output_dir, loaded
        )
        lightweight_resume_state = _metadata_trainer_state(resume_checkpoint)

        model_name = _resolve_model_name(
            model_config.get("base_model", DEFAULT_MODEL_NAME), loaded
        )
        try:
            from transformers import ChineseCLIPProcessor
        except ImportError as exc:
            raise RuntimeError("训练需要安装 transformers") from exc
        processor = ChineseCLIPProcessor.from_pretrained(model_name)
        model = load_chinese_clip_model(
            model_name,
            use_sdpa=bool(model_config.get("use_sdpa", True)),
        )
        model = apply_dual_tower_lora(
            model,
            rank=int(lora_config.get("rank", 8)),
            alpha=int(lora_config.get("alpha", 16)),
            dropout=float(lora_config.get("dropout", 0.05)),
            verify_expected_count=bool(model_config.get("verify_expected_trainable", True)),
        )
        if bool(model_config.get("gradient_checkpointing", True)):
            enable_gradient_checkpointing(model)
        model.to(device)
        model.train()

        dataset = _build_training_dataset(data_config, loaded)
        collator = MultiPositiveCollator(
            processor,
            dataset.text_to_images,
            dataset.image_to_texts,
            max_length=int(data_config.get("max_length", 52)),
            padding="max_length",
        )

        configured_micro = training_config.get("micro_batch_size", "auto")
        candidates = training_config.get(
            "micro_batch_candidates", DEFAULT_MICRO_BATCH_CANDIDATES
        )
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("training.micro_batch_candidates 必须是整数列表")
        saved_micro = lightweight_resume_state.get("micro_batch_size")
        if saved_micro is not None:
            micro_batch_size = int(saved_micro)
            probes: tuple[MicroBatchProbe, ...] = ()
        else:
            micro_batch_size, probes = select_micro_batch_size(
                configured_micro,
                [int(value) for value in candidates],
                memory_fraction=memory_fraction,
                probe=_probe_callable(dataset, collator, model, device, policy),
                total_memory_bytes=total_memory,
            )
        for result in probes:
            reporter.record("micro_batch_probe", **asdict(result))

        accumulation_steps = compute_accumulation_steps(
            effective_batch_size, micro_batch_size
        )
        batches_per_epoch = math.ceil(epoch_size / micro_batch_size)
        optimizer_steps_per_epoch = len(
            accumulation_group_sizes(batches_per_epoch, accumulation_steps)
        )
        total_optimizer_steps = epochs * optimizer_steps_per_epoch

        sampler = TemperatureBalancedSampler(
            dataset,
            epoch_size=epoch_size,
            seed=seed,
            exponent=float(data_config.get("temperature_exponent", 0.5)),
        )
        parameter_groups = split_trainable_parameter_groups(
            model,
            lora_lr=float(optimizer_config.get("lora_lr", 1e-4)),
            projection_lr=float(optimizer_config.get("projection_lr", 5e-5)),
            logit_scale_lr=float(optimizer_config.get("logit_scale_lr", 1e-5)),
            projection_weight_decay=float(
                optimizer_config.get("projection_weight_decay", 1e-3)
            ),
        )
        optimizer = torch.optim.AdamW(parameter_groups)
        scheduler, warmup_steps = build_cosine_scheduler(
            optimizer,
            total_optimizer_steps,
            float(optimizer_config.get("warmup_ratio", 0.05)),
        )
        scaler = _make_grad_scaler(policy)

        trainer_state: dict[str, Any] = {
            "epoch": 0,
            "optimizer_step": 0,
            "micro_batches_consumed": 0,
            "samples_consumed": 0,
            "best_score": None,
            "best_checkpoint": None,
            "bad_validations": 0,
            "last_eval_step": -1,
            "elapsed_seconds": 0.0,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "total_optimizer_steps": total_optimizer_steps,
            "epochs": epochs,
        }
        if resume_checkpoint is not None:
            restored = load_training_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                device=device,
                restore_rng=True,
            )
            restored_state = dict(restored["trainer_state"])
            for key, current in (
                ("micro_batch_size", micro_batch_size),
                ("accumulation_steps", accumulation_steps),
                ("effective_batch_size", effective_batch_size),
                ("total_optimizer_steps", total_optimizer_steps),
                ("epochs", epochs),
            ):
                if key in restored_state and int(restored_state[key]) != int(current):
                    raise ValueError(
                        f"恢复配置不兼容：checkpoint {key}={restored_state[key]}，"
                        f"当前为 {current}"
                    )
            trainer_state.update(restored_state)

        report = parameter_report(model)
        start_monotonic = time.monotonic()
        elapsed_before_resume = float(trainer_state.get("elapsed_seconds", 0.0))
        samples_at_start = int(trainer_state.get("samples_consumed", 0))
        early_stopped = int(trainer_state.get("bad_validations", 0)) >= patience
        last_loss: float | None = None

        reporter.record(
            "training_started" if resume_checkpoint is None else "training_resumed",
            status="running",
            config_path=loaded.get("_config_path"),
            resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
            device=str(device),
            precision=policy.name,
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
            effective_batch_size=effective_batch_size,
            epochs=epochs,
            epoch_size=epoch_size,
            optimizer_step=int(trainer_state["optimizer_step"]),
            total_optimizer_steps=total_optimizer_steps,
            warmup_steps=warmup_steps,
            trainable_parameters=report.trainable,
            trainable_percentage=report.percentage,
            memory_fraction_limit=memory_fraction,
            **_gpu_metrics(device, total_memory),
        )

        def current_elapsed() -> float:
            return elapsed_before_resume + (time.monotonic() - start_monotonic)

        def state_for_checkpoint(epoch_index: int) -> dict[str, Any]:
            state = dict(trainer_state)
            state["epoch"] = int(epoch_index)
            state["elapsed_seconds"] = current_elapsed()
            return state

        def validate_and_checkpoint(epoch_index: int) -> bool:
            nonlocal early_stopped
            step = int(trainer_state["optimizer_step"])
            if int(trainer_state.get("last_eval_step", -1)) == step:
                return early_stopped
            validation_started = time.monotonic()
            LOGGER.info(
                "开始全量 valid：epoch=%d step=%d/%d",
                epoch_index + 1,
                step,
                total_optimizer_steps,
            )
            reporter.record(
                "validation_started",
                status="evaluating",
                epoch=epoch_index + 1,
                optimizer_step=step,
                total_optimizer_steps=total_optimizer_steps,
                loss=last_loss,
                lr=float(optimizer.param_groups[0]["lr"]),
                learning_rates=_learning_rates(optimizer),
                best_score=trainer_state.get("best_score"),
                best_checkpoint=trainer_state.get("best_checkpoint"),
                **_gpu_metrics(device, total_memory),
            )
            validation = dict(
                evaluator(model, processor, loaded, split="valid", device=device)
            )
            score = validation.get("selection_score")
            if not bool(validation.get("eligible_for_selection")) or score is None:
                raise RuntimeError(
                    "全量 valid 未产生固定五任务宏平均，不能用于 best checkpoint 选择"
                )
            score = float(score)
            if not math.isfinite(score):
                raise FloatingPointError(f"验证 selection_score 非有限值：{score}")
            previous_best = trainer_state.get("best_score")
            improved = previous_best is None or score > float(previous_best)
            checkpoint_dir = output_dir / "checkpoints" / f"step-{step:08d}"
            if improved:
                trainer_state["best_score"] = score
                trainer_state["best_checkpoint"] = str(checkpoint_dir.resolve())
                trainer_state["bad_validations"] = 0
            else:
                trainer_state["bad_validations"] = int(
                    trainer_state.get("bad_validations", 0)
                ) + 1
            trainer_state["last_eval_step"] = step
            checkpoint_state = state_for_checkpoint(epoch_index)
            if checkpoint_dir.exists():
                raise FileExistsError(f"checkpoint 已存在，拒绝覆盖：{checkpoint_dir}")
            save_training_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                sampler=sampler,
                trainer_state=checkpoint_state,
                metadata={
                    "base_model": model_name,
                    "precision": policy.name,
                    "validation_score": score,
                    "is_best": improved,
                    "config": loaded,
                },
            )
            if improved:
                _atomic_json(
                    output_dir / "best_checkpoint.json",
                    {
                        "checkpoint_dir": str(checkpoint_dir.resolve()),
                        "optimizer_step": step,
                        "selection_score": score,
                    },
                )
            early_stopped = int(trainer_state["bad_validations"]) >= patience
            remaining_steps = max(total_optimizer_steps - step, 0)
            elapsed = current_elapsed()
            completed_this_run = max(step - int(lightweight_resume_state.get("optimizer_step", 0)), 1)
            eta = remaining_steps * max(time.monotonic() - start_monotonic, 0.0) / completed_this_run
            reporter.record(
                "validation",
                status="early_stopped" if early_stopped else "running",
                epoch=epoch_index + 1,
                optimizer_step=step,
                loss=last_loss,
                lr=float(optimizer.param_groups[0]["lr"]),
                learning_rates=_learning_rates(optimizer),
                validation=validation,
                validation_score=score,
                validation_seconds=time.monotonic() - validation_started,
                best_score=trainer_state["best_score"],
                best_checkpoint=trainer_state["best_checkpoint"],
                bad_validations=trainer_state["bad_validations"],
                early_stopping_patience=patience,
                eta_seconds=eta,
                elapsed_seconds=elapsed,
                **_gpu_metrics(device, total_memory),
            )
            LOGGER.info(
                "完成全量 valid：epoch=%d step=%d/%d score=%.6f best=%.6f "
                "bad_validations=%d/%d status=%s",
                epoch_index + 1,
                step,
                total_optimizer_steps,
                score,
                float(trainer_state["best_score"]),
                int(trainer_state["bad_validations"]),
                patience,
                "early_stopped" if early_stopped else "running",
            )
            return early_stopped

        epoch = int(trainer_state.get("epoch", 0))
        while epoch < epochs and not early_stopped:
            if sampler.epoch != epoch:
                sampler.set_epoch(epoch)
            if sampler.remaining == 0:
                epoch += 1
                if epoch < epochs:
                    sampler.set_epoch(epoch)
                trainer_state["epoch"] = epoch
                continue

            loader = _make_loader(
                dataset,
                sampler,
                collator,
                batch_size=micro_batch_size,
                num_workers=num_workers,
                seed=seed,
            )
            iterator = iter(loader)
            remaining_micro_batches = math.ceil(sampler.remaining / micro_batch_size)
            group_sizes = accumulation_group_sizes(
                remaining_micro_batches, accumulation_steps
            )
            for group_size in group_sizes:
                optimizer.zero_grad(set_to_none=True)
                group_loss = 0.0
                group_text_loss = 0.0
                group_image_loss = 0.0
                group_samples = 0
                for _ in range(group_size):
                    try:
                        raw_batch = next(iterator)
                    except StopIteration as exc:
                        raise RuntimeError(
                            "DataLoader 提前结束，sampler 游标与 batch 计划不一致"
                        ) from exc
                    actual_batch_size = int(raw_batch["positive_mask"].shape[0])
                    sampler.advance(actual_batch_size)
                    trainer_state["micro_batches_consumed"] = int(
                        trainer_state["micro_batches_consumed"]
                    ) + 1
                    trainer_state["samples_consumed"] = int(
                        trainer_state["samples_consumed"]
                    ) + actual_batch_size
                    group_samples += actual_batch_size
                    batch = _move_batch(raw_batch, device)
                    with _autocast_context(device, policy):
                        text_features, image_features = encode_training_batch(model, batch)
                        loss, text_loss, image_loss = symmetric_multi_positive_loss(
                            text_features,
                            image_features,
                            batch["positive_mask"],
                            get_base_chinese_clip(model).logit_scale,
                            return_directional=True,
                        )
                    if not bool(torch.isfinite(loss.detach()).item()):
                        raise FloatingPointError(
                            f"optimizer step {trainer_state['optimizer_step']} 出现非有限 loss"
                        )
                    scaled_loss = loss / float(group_size)
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    group_loss += float(loss.detach().item())
                    group_text_loss += float(text_loss.detach().item())
                    group_image_loss += float(image_loss.detach().item())

                if scaler is not None:
                    scaler.unscale_(optimizer)
                finite, parameter_name = trainable_gradients_are_finite(model)
                if not finite:
                    raise FloatingPointError(
                        f"发现非有限梯度，首个参数为 {parameter_name}"
                    )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=gradient_clip,
                )
                if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                    raise FloatingPointError(f"梯度范数非有限值：{grad_norm}")
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                clamp_logit_scale_(model, maximum=100.0)
                scheduler.step()
                trainer_state["optimizer_step"] = int(
                    trainer_state["optimizer_step"]
                ) + 1
                step = int(trainer_state["optimizer_step"])
                last_loss = group_loss / float(group_size)
                elapsed_run = max(time.monotonic() - start_monotonic, 1e-9)
                run_samples = int(trainer_state["samples_consumed"]) - samples_at_start
                throughput = run_samples / elapsed_run
                completed_run_steps = max(
                    step - int(lightweight_resume_state.get("optimizer_step", 0)), 1
                )
                eta = (
                    max(total_optimizer_steps - step, 0)
                    * elapsed_run
                    / completed_run_steps
                )
                rates = _learning_rates(optimizer)
                memory_values = _gpu_metrics(device, total_memory)
                reporter.record(
                    "train_step",
                    status="running",
                    epoch=epoch + 1,
                    epoch_sample_position=sampler.position,
                    optimizer_step=step,
                    total_optimizer_steps=total_optimizer_steps,
                    micro_batches_in_group=group_size,
                    samples_in_group=group_samples,
                    samples_consumed=trainer_state["samples_consumed"],
                    loss=last_loss,
                    text_to_image_loss=group_text_loss / float(group_size),
                    image_to_text_loss=group_image_loss / float(group_size),
                    grad_norm=float(torch.as_tensor(grad_norm).detach().cpu().item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    learning_rates=rates,
                    logit_scale=float(
                        get_base_chinese_clip(model).logit_scale.detach().float().cpu().item()
                    ),
                    throughput_samples_per_second=throughput,
                    eta_seconds=eta,
                    elapsed_seconds=current_elapsed(),
                    best_score=trainer_state.get("best_score"),
                    best_checkpoint=trainer_state.get("best_checkpoint"),
                    **memory_values,
                )
                LOGGER.info(
                    "epoch=%d step=%d/%d loss=%.6f lr=%.3e throughput=%.2f "
                    "samples/s gpu_peak=%.2fGiB eta=%.0fs",
                    epoch + 1,
                    step,
                    total_optimizer_steps,
                    last_loss,
                    float(optimizer.param_groups[0]["lr"]),
                    throughput,
                    float(memory_values["gpu_memory_peak_bytes"]) / (1024.0**3),
                    eta,
                )
                if step % eval_every == 0 and validate_and_checkpoint(epoch):
                    break

            del iterator, loader
            if early_stopped:
                break
            if int(trainer_state.get("last_eval_step", -1)) != int(
                trainer_state["optimizer_step"]
            ):
                if validate_and_checkpoint(epoch):
                    break
            epoch += 1
            trainer_state["epoch"] = epoch
            if epoch < epochs:
                sampler.set_epoch(epoch)

        final_status = "early_stopped" if early_stopped else "completed"
        trainer_state["elapsed_seconds"] = current_elapsed()
        final_payload = {
            "status": final_status,
            "epoch": min(epoch + 1, epochs) if early_stopped else min(epoch, epochs),
            "optimizer_step": int(trainer_state["optimizer_step"]),
            "total_optimizer_steps": total_optimizer_steps,
            "loss": last_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "learning_rates": _learning_rates(optimizer),
            "best_score": trainer_state.get("best_score"),
            "best_checkpoint": trainer_state.get("best_checkpoint"),
            "samples_consumed": int(trainer_state["samples_consumed"]),
            "elapsed_seconds": trainer_state["elapsed_seconds"],
            "throughput_samples_per_second": (
                (int(trainer_state["samples_consumed"]) - samples_at_start)
                / max(time.monotonic() - start_monotonic, 1e-9)
            ),
            "eta_seconds": 0.0,
            **_gpu_metrics(device, total_memory),
        }
        reporter.record("training_finished", **final_payload)
        return final_payload
    except BaseException as exc:
        reporter.mark_failed(exc)
        raise
    finally:
        _close_dataset(dataset)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """读取 YAML 训练入口参数；命令行 resume 优先于 YAML。"""

    parser = argparse.ArgumentParser(description="微调 Chinese-CLIP 双塔 LoRA")
    parser.add_argument("--config", required=True, help="训练 YAML 配置路径")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="恢复 checkpoint 路径；不带值时恢复 output_dir 中最新 checkpoint",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行执行训练，最终摘要输出为一行 JSON。"""

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    result = train(args.config, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_MICRO_BATCH_CANDIDATES",
    "MicroBatchProbe",
    "PrecisionPolicy",
    "accumulation_group_sizes",
    "build_cosine_scheduler",
    "build_train_image_transform",
    "compute_accumulation_steps",
    "cosine_warmup_multiplier",
    "load_yaml_config",
    "main",
    "parse_args",
    "resolve_resume_checkpoint",
    "resolve_training_precision",
    "select_micro_batch_size",
    "train",
]
