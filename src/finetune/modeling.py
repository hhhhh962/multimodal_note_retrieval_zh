"""Model construction helpers for dual-tower Chinese-CLIP LoRA training.

The public functions in this module deliberately keep Transformers and PEFT
loading behind function calls.  This makes configuration and unit tests usable
without downloading a model, while the training process still uses the normal
Hugging Face ``ChineseCLIPModel`` format.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
TEXT_LORA_TARGETS = ("query", "key", "value")
VISION_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj")
LORA_TARGET_MODULES = TEXT_LORA_TARGETS + VISION_LORA_TARGETS

DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
EXPECTED_TRAINABLE_PARAMETERS = 1_818_625
EXPECTED_WRAPPED_PARAMETERS = 190_081_537


@dataclass(frozen=True)
class TrainableParameterReport:
    """Summary returned after the LoRA model has been assembled."""

    trainable: int
    total: int
    percentage: float
    trainable_names: tuple[str, ...]


def _load_model_once(
    model_cls: type,
    model_name_or_path: str,
    *,
    attention_implementation: str,
    model_kwargs: Mapping[str, Any],
) -> nn.Module:
    kwargs = dict(model_kwargs)
    kwargs["attn_implementation"] = attention_implementation
    return model_cls.from_pretrained(model_name_or_path, **kwargs)


def load_chinese_clip_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    *,
    use_sdpa: bool = True,
    model_cls: type | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    """Load Chinese-CLIP, preferring SDPA and falling back to eager attention.

    The fallback is intentionally limited to compatibility-style failures.  A
    missing checkpoint, authentication error, or network error is not hidden by
    a second download attempt.

    ``model_cls`` is injectable so the fallback behaviour can be tested without
    network access.
    """

    if model_cls is None:
        from transformers import ChineseCLIPModel

        model_cls = ChineseCLIPModel

    implementation = "sdpa" if use_sdpa else "eager"
    try:
        return _load_model_once(
            model_cls,
            model_name_or_path,
            attention_implementation=implementation,
            model_kwargs=model_kwargs,
        )
    except (TypeError, ValueError, NotImplementedError) as exc:
        if implementation != "sdpa":
            raise
        LOGGER.warning(
            "SDPA is not compatible with this Chinese-CLIP/runtime (%s); "
            "falling back to eager attention.",
            exc,
        )
        return _load_model_once(
            model_cls,
            model_name_or_path,
            attention_implementation="eager",
            model_kwargs=model_kwargs,
        )


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable non-reentrant checkpointing when the installed version supports it."""

    target = get_base_chinese_clip(model)
    method = getattr(target, "gradient_checkpointing_enable", None)
    if method is None:
        raise AttributeError("Chinese-CLIP model does not expose gradient_checkpointing_enable()")
    try:
        method(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        method()

    # PEFT exposes this helper on recent releases.  It is harmless for ordinary
    # LoRA and protects older re-entrant checkpoint implementations whose first
    # inputs otherwise do not require gradients.
    input_grad_method = getattr(model, "enable_input_require_grads", None)
    if input_grad_method is not None:
        input_grad_method()


def get_base_chinese_clip(model: nn.Module) -> nn.Module:
    """Return the underlying Hugging Face ChineseCLIPModel from a PEFT wrapper."""

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        return get_base_model()

    candidate = model
    # Accommodate generic wrappers without depending on a particular PEFT class.
    for _ in range(3):
        if all(hasattr(candidate, name) for name in ("text_projection", "visual_projection", "logit_scale")):
            return candidate
        next_candidate = getattr(candidate, "model", None)
        if next_candidate is None or next_candidate is candidate:
            break
        candidate = next_candidate
    return candidate


def get_projection_module(model: nn.Module, name: str) -> nn.Module:
    """Return the active projection, unwrapping PEFT ModulesToSaveWrapper.

    Adding the projections to ``modules_to_save`` makes the adapter directory
    self-contained.  PEFT keeps a frozen original plus an active trainable copy;
    checkpoint code must serialize the latter, not the wrapper's combined state.
    """

    if name not in {"text_projection", "visual_projection"}:
        raise ValueError(f"Unknown projection module: {name}")
    module = getattr(get_base_chinese_clip(model), name, None)
    if module is None:
        raise AttributeError(f"Chinese-CLIP model is missing {name}")
    modules_to_save = getattr(module, "modules_to_save", None)
    if modules_to_save is None:
        return module
    active = getattr(module, "active_adapter", "default")
    if isinstance(active, (tuple, list)):
        if len(active) != 1:
            raise ValueError(f"Expected one active projection adapter, got {active!r}")
        active = active[0]
    if active not in modules_to_save:
        raise KeyError(f"Projection wrapper has no active adapter {active!r}")
    return modules_to_save[active]


def apply_dual_tower_lora(
    model: nn.Module,
    *,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    verify_expected_count: bool = True,
    expected_trainable: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> nn.Module:
    """Freeze Chinese-CLIP, attach both-tower LoRA, and unfreeze three extras.

    Text attention adapts ``query/key/value``.  Vision attention adapts
    ``q_proj/k_proj/v_proj/out_proj``.  The projection heads and scalar
    ``logit_scale`` remain directly trainable.
    """

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=["text_projection", "visual_projection"],
    )
    peft_model = get_peft_model(model, lora_config)
    base_model = get_base_chinese_clip(peft_model)

    for name in ("text_projection", "visual_projection"):
        module = get_projection_module(peft_model, name)
        if not any(parameter.requires_grad for parameter in module.parameters()):
            raise AssertionError(f"PEFT did not make {name} trainable via modules_to_save")

    logit_scale = getattr(base_model, "logit_scale", None)
    if not isinstance(logit_scale, nn.Parameter):
        raise TypeError("Chinese-CLIP logit_scale must be an nn.Parameter")
    logit_scale.requires_grad_(True)

    assert_only_expected_parameters_trainable(peft_model)
    if verify_expected_count:
        assert_expected_trainable_count(peft_model, expected_trainable)
    return peft_model


def parameter_report(model: nn.Module) -> TrainableParameterReport:
    trainable_names: list[str] = []
    trainable = 0
    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
    percentage = 100.0 * trainable / total if total else 0.0
    return TrainableParameterReport(
        trainable=trainable,
        total=total,
        percentage=percentage,
        trainable_names=tuple(trainable_names),
    )


def _is_allowed_trainable_name(name: str) -> bool:
    return (
        "lora_A" in name
        or "lora_B" in name
        or ".text_projection." in name
        or name.startswith("text_projection.")
        or ".visual_projection." in name
        or name.startswith("visual_projection.")
        or name.endswith("logit_scale")
    )


def assert_only_expected_parameters_trainable(model: nn.Module) -> None:
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_allowed_trainable_name(name)
    ]
    if unexpected:
        preview = ", ".join(unexpected[:10])
        raise AssertionError(f"Unexpected trainable parameters: {preview}")


def assert_expected_trainable_count(
    model: nn.Module,
    expected: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> TrainableParameterReport:
    report = parameter_report(model)
    if report.trainable != expected:
        raise AssertionError(
            "Trainable parameter count mismatch: "
            f"expected {expected:,}, got {report.trainable:,}. "
            "Check the base checkpoint architecture and PEFT target modules."
        )
    return report


def split_trainable_parameter_groups(
    model: nn.Module,
    *,
    lora_lr: float = 1e-4,
    projection_lr: float = 5e-5,
    logit_scale_lr: float = 1e-5,
    projection_weight_decay: float = 1e-3,
) -> list[MutableMapping[str, Any]]:
    """Build the exact three AdamW groups required by the training plan."""

    lora_parameters: list[nn.Parameter] = []
    projection_parameters: list[nn.Parameter] = []
    temperature_parameters: list[nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_parameters.append(parameter)
        elif "text_projection" in name or "visual_projection" in name:
            projection_parameters.append(parameter)
        elif name.endswith("logit_scale"):
            temperature_parameters.append(parameter)
        else:
            raise AssertionError(f"Trainable parameter was not assigned to an optimizer group: {name}")

    if not lora_parameters or not projection_parameters or len(temperature_parameters) != 1:
        raise AssertionError(
            "Expected non-empty LoRA/projection groups and exactly one logit_scale parameter"
        )

    return [
        {
            "params": lora_parameters,
            "lr": float(lora_lr),
            "weight_decay": 0.0,
            "group_name": "lora",
        },
        {
            "params": projection_parameters,
            "lr": float(projection_lr),
            "weight_decay": float(projection_weight_decay),
            "group_name": "projections",
        },
        {
            "params": temperature_parameters,
            "lr": float(logit_scale_lr),
            "weight_decay": 0.0,
            "group_name": "logit_scale",
        },
    ]


def encode_training_batch(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return text and image embeddings from one paired training batch."""

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pixel_values=batch["pixel_values"],
        return_dict=True,
    )
    text_features = getattr(outputs, "text_embeds", None)
    image_features = getattr(outputs, "image_embeds", None)
    if text_features is None or image_features is None:
        raise RuntimeError("Chinese-CLIP forward output is missing text_embeds/image_embeds")
    return text_features, image_features


def clamp_logit_scale_(model: nn.Module, maximum: float = 100.0) -> None:
    """Clamp the learned log temperature to ``[0, log(maximum)]`` in place."""

    if maximum <= 1.0:
        raise ValueError("maximum logit scale must be greater than one")
    upper = torch.log(torch.tensor(maximum)).item()
    base_model = get_base_chinese_clip(model)
    with torch.no_grad():
        base_model.logit_scale.clamp_(0.0, upper)


def trainable_gradients_are_finite(model: nn.Module) -> tuple[bool, str | None]:
    """Check every trainable tensor; missing gradients are also a hard failure."""

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            return False, name
        if not bool(torch.isfinite(parameter.grad).all()):
            return False, name
    return True, None


def merge_lora_for_export(model: nn.Module) -> nn.Module:
    """Merge adapters in memory and return a plain ChineseCLIPModel."""

    merge = getattr(model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("Model is not a mergeable PEFT model")
    merged = merge(safe_merge=True)
    assert not hasattr(merged, "peft_config"), "PEFT wrapper remained after merge_and_unload"
    return merged


__all__ = [
    "DEFAULT_MODEL_NAME",
    "EXPECTED_TRAINABLE_PARAMETERS",
    "EXPECTED_WRAPPED_PARAMETERS",
    "LORA_TARGET_MODULES",
    "TrainableParameterReport",
    "apply_dual_tower_lora",
    "assert_expected_trainable_count",
    "assert_only_expected_parameters_trainable",
    "clamp_logit_scale_",
    "enable_gradient_checkpointing",
    "encode_training_batch",
    "get_base_chinese_clip",
    "get_projection_module",
    "load_chinese_clip_model",
    "merge_lora_for_export",
    "parameter_report",
    "split_trainable_parameter_groups",
    "trainable_gradients_are_finite",
]
