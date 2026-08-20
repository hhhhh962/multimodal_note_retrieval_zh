"""把训练 checkpoint 合并并导出成现有搜索代码可直接加载的 HF 模型。

训练阶段保留 PEFT adapter 便于恢复；部署阶段调用 ``merge_and_unload``，将 LoRA 权重
写回普通 ``ChineseCLIPModel``。导出成功的必要条件是本地重新加载后文本塔和视觉塔均
输出有限的 512 维特征。最终目录同时包含 processor、内容指纹和导出清单。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from model_fingerprint import write_model_fingerprint
except ImportError:
    from src.model_fingerprint import write_model_fingerprint


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """读取 YAML/JSON 配置或复制调用方传入的 Mapping。"""

    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取 YAML 配置需要安装 pyyaml") from exc
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"配置根节点必须是对象：{path}")
    return dict(loaded)


def _base_chinese_clip(model: Any) -> Any:
    """从 PEFT 包装中找到实际 ChineseCLIPModel，同时保留已注入的 LoRA 层。"""

    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        candidate = getter()
        if candidate is not None:
            return candidate
    candidate = getattr(getattr(model, "base_model", None), "model", None)
    return candidate if candidate is not None else model


def _projection_module(model: Any, name: str) -> Any:
    """复用训练建模工具解析 modules_to_save 包装后的投影层。"""

    try:
        from src.finetune.modeling import get_projection_module
    except ImportError:
        try:
            from finetune.modeling import get_projection_module
        except ImportError:
            get_projection_module = None
    if get_projection_module is not None:
        try:
            return get_projection_module(model, name)
        except TypeError:
            return get_projection_module(model, name.replace("_projection", ""))
    base = _base_chinese_clip(model)
    module = getattr(base, name, None)
    if module is None:
        raise AttributeError(f"模型中找不到 {name}")
    active = getattr(module, "active_adapter", None)
    saved = getattr(module, "modules_to_save", None)
    if saved is not None and active is not None:
        adapter_name = active[0] if isinstance(active, (list, tuple)) else active
        if adapter_name in saved:
            return saved[adapter_name]
    return module


def _restore_extra_trainable_state(model: Any, checkpoint_dir: Path) -> None:
    """调用训练侧统一恢复函数；仅在旧 checkpoint 下使用兼容兜底。"""

    state_path = checkpoint_dir / "trainable_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"缺少额外可训练参数：{state_path}")
    import torch

    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(state_path, map_location="cpu")
    restore = None
    try:
        from src.finetune.checkpointing import restore_extra_trainable_state as restore
    except ImportError:
        try:
            from finetune.checkpointing import restore_extra_trainable_state as restore
        except ImportError:
            restore = None
    if restore is not None:
        restore(model, state)
        return
    _projection_module(model, "text_projection").load_state_dict(state["text_projection"])
    _projection_module(model, "visual_projection").load_state_dict(state["visual_projection"])
    base = _base_chinese_clip(model)
    with torch.no_grad():
        base.logit_scale.copy_(state["logit_scale"].to(base.logit_scale.device))


def merge_lora(model: Any) -> Any:
    """将 LoRA 合并回普通模型；若输入不是 PEFT 模型则拒绝静默导出。"""

    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("待导出对象不支持 merge_and_unload，无法确认 LoRA 已合并")
    merged = merge(safe_merge=True)
    if merged is None:
        raise RuntimeError("merge_and_unload 未返回合并后的模型")
    return merged


def _feature_tensor(value: Any, preferred: str) -> Any:
    """兼容不同 Transformers 版本，将双塔特征输出统一解包为张量。"""

    import torch

    if isinstance(value, torch.Tensor):
        return value
    for attr in (preferred, "text_embeds", "image_embeds", "pooler_output"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"无法从 {type(value)!r} 提取特征张量")


def verify_exported_model(model_dir: str | Path, expected_dimension: int = 512) -> dict[str, Any]:
    """完全从磁盘重载普通 ChineseCLIPModel，并验证双塔输出维度与有限性。"""

    import numpy as np
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    path = Path(model_dir)
    processor = ChineseCLIPProcessor.from_pretrained(path, local_files_only=True)
    model = ChineseCLIPModel.from_pretrained(path, local_files_only=True).eval()
    text_inputs = processor(
        text=["合并模型验证"],
        padding=True,
        truncation=True,
        max_length=52,
        return_tensors="pt",
    )
    image = Image.new("RGB", (224, 224), color=(127, 127, 127))
    try:
        image_inputs = processor(images=[image], return_tensors="pt")
    finally:
        image.close()
    with torch.inference_mode():
        text_features = _feature_tensor(
            model.get_text_features(**text_inputs),
            preferred="text_embeds",
        )
        image_features = _feature_tensor(
            model.get_image_features(**image_inputs),
            preferred="image_embeds",
        )
    text_shape = tuple(int(value) for value in text_features.shape)
    image_shape = tuple(int(value) for value in image_features.shape)
    expected_shape = (1, int(expected_dimension))
    if text_shape != expected_shape or image_shape != expected_shape:
        raise ValueError(
            f"导出模型维度错误：text={text_shape}, image={image_shape}, expected={expected_shape}"
        )
    if not torch.isfinite(text_features).all() or not torch.isfinite(image_features).all():
        raise ValueError("导出模型产生 NaN 或 Inf 特征")
    return {
        "loader": "transformers.ChineseCLIPModel.from_pretrained",
        "text_shape": list(text_shape),
        "image_shape": list(image_shape),
        "dimension": int(expected_dimension),
        "text_l2_norm": float(np.linalg.norm(text_features.float().cpu().numpy())),
        "image_l2_norm": float(np.linalg.norm(image_features.float().cpu().numpy())),
    }


def export_merged_model(
    model: Any,
    processor: Any,
    output_dir: str | Path,
    source_checkpoint: str | Path | None = None,
    base_model: str | None = None,
    data_manifest_hash: str | None = None,
    expected_dimension: int = 512,
    verify_reload: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """合并、保存、重载验证并写入指纹；失败时不留下半成品目标目录。"""

    target = Path(output_dir).resolve()
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"导出目录非空，拒绝覆盖：{target}")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    try:
        merged = merge_lora(model)
        if not hasattr(merged, "save_pretrained"):
            raise TypeError("merge_and_unload 返回对象不支持 save_pretrained")
        merged.save_pretrained(temporary, safe_serialization=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.model_max_length = 52
        processor.save_pretrained(temporary)
        retrieval_config = {
            "max_length": 52,
            "dimension": int(expected_dimension),
            "normalization": "l2",
        }
        (temporary / "retrieval_config.json").write_text(
            json.dumps(retrieval_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        verification = (
            verify_exported_model(temporary, expected_dimension=expected_dimension)
            if verify_reload
            else {"skipped": True, "dimension": int(expected_dimension)}
        )
        fingerprint = write_model_fingerprint(temporary)
        manifest = {
            "format": "huggingface_chinese_clip_merged_lora",
            "load_with": "transformers.ChineseCLIPModel.from_pretrained",
            "source_checkpoint": str(Path(source_checkpoint).resolve()) if source_checkpoint else None,
            "base_model": base_model,
            "model_fingerprint": fingerprint["value"],
            "data_manifest_hash": data_manifest_hash,
            "dimension": int(expected_dimension),
            "normalization": "l2",
            "max_length": 52,
            "verification": verification,
        }
        (temporary / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            target.rmdir()
        temporary.replace(target)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint_for_export(
    checkpoint_dir: str | Path,
    base_model_name: str,
    processor_name: str | None = None,
    device: str = "cpu",
) -> tuple[Any, Any]:
    """加载基座、adapter 与额外投影/温度状态，为合并导出做好准备。"""

    import torch
    from peft import PeftModel
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    checkpoint = Path(checkpoint_dir)
    adapter_dir = checkpoint / "adapter"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"缺少 PEFT adapter 目录：{adapter_dir}")
    base = ChineseCLIPModel.from_pretrained(base_model_name)
    processor = ChineseCLIPProcessor.from_pretrained(processor_name or base_model_name)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    _restore_extra_trainable_state(model, checkpoint)
    model.to(torch.device(device)).eval()
    return model, processor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """读取 checkpoint 合并导出参数。"""

    parser = argparse.ArgumentParser(description="合并 Chinese-CLIP 双塔 LoRA 并导出完整 HF 模型")
    parser.add_argument("--config", default="configs/finetune_lora.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_manifest_hash", default=None)
    parser.add_argument("--expected_dimension", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_verify_reload", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """从训练 checkpoint 导出最终部署模型。"""

    args = parse_args(argv)
    config = _load_config(args.config)
    model_config = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    base_model = args.base_model or model_config.get("base_model") or model_config.get("name")
    if not base_model:
        base_model = "OFA-Sys/chinese-clip-vit-base-patch16"
    model, processor = load_checkpoint_for_export(
        checkpoint_dir=args.checkpoint,
        base_model_name=str(base_model),
        processor_name=args.processor,
        device=args.device,
    )
    manifest = export_merged_model(
        model=model,
        processor=processor,
        output_dir=args.output_dir,
        source_checkpoint=args.checkpoint,
        base_model=str(base_model),
        data_manifest_hash=args.data_manifest_hash,
        expected_dimension=args.expected_dimension,
        verify_reload=not args.no_verify_reload,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
