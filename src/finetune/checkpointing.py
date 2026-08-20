"""Checkpoint and machine-readable progress utilities for LoRA fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, MutableMapping
from uuid import uuid4

import numpy as np
import torch
from torch import nn

from .modeling import get_base_chinese_clip, get_projection_module


ADAPTER_DIRNAME = "adapter"
TRAINABLE_STATE_FILENAME = "trainable_state.pt"
TRAINER_STATE_FILENAME = "trainer_state.pt"
METADATA_FILENAME = "metadata.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(payload), stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class ProgressReporter:
    """Write a current snapshot plus an append-only metric/event stream.

    ``progress.json`` is intended for ``watch``/dashboard consumption.  Every
    event is also retained in ``metrics.jsonl`` for plotting and post-mortems.
    Human-readable logging remains on stdout/stderr so the launcher can redirect
    it to ``train.log``.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_dir / "progress.json"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.snapshot: dict[str, Any] = {
            "status": "initializing",
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
        _atomic_write_json(self.progress_path, self.snapshot)

    def record(self, event: str, /, **values: Any) -> Mapping[str, Any]:
        timestamp = utc_now()
        row = {"timestamp": timestamp, "event": event, **_json_safe(values)}
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        self.snapshot.update(_json_safe(values))
        self.snapshot["last_event"] = event
        self.snapshot["updated_at"] = timestamp
        _atomic_write_json(self.progress_path, self.snapshot)
        return row

    def mark_failed(self, error: BaseException) -> None:
        self.record(
            "failed",
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _state_dict_on_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def collect_extra_trainable_state(model: nn.Module) -> dict[str, Any]:
    base_model = get_base_chinese_clip(model)
    return {
        "text_projection": _state_dict_on_cpu(get_projection_module(model, "text_projection")),
        "visual_projection": _state_dict_on_cpu(get_projection_module(model, "visual_projection")),
        "logit_scale": base_model.logit_scale.detach().cpu().clone(),
    }


def restore_extra_trainable_state(model: nn.Module, state: Mapping[str, Any]) -> None:
    base_model = get_base_chinese_clip(model)
    get_projection_module(model, "text_projection").load_state_dict(
        state["text_projection"], strict=True
    )
    get_projection_module(model, "visual_projection").load_state_dict(
        state["visual_projection"], strict=True
    )
    saved_temperature = state["logit_scale"]
    with torch.no_grad():
        base_model.logit_scale.copy_(
            saved_temperature.to(
                device=base_model.logit_scale.device,
                dtype=base_model.logit_scale.dtype,
            )
        )


def _sampler_state_dict(sampler: Any | None) -> Mapping[str, Any] | None:
    if sampler is None:
        return None
    method = getattr(sampler, "state_dict", None)
    if callable(method):
        return method()
    return {
        key: getattr(sampler, key)
        for key in ("epoch", "position")
        if hasattr(sampler, key)
    }


def save_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any | None,
    sampler: Any | None,
    trainer_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save adapter, extra trainables, optimizer, and exact RNG state."""

    destination = Path(checkpoint_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=False, exist_ok=False)

    try:
        save_pretrained = getattr(model, "save_pretrained", None)
        if not callable(save_pretrained):
            raise TypeError("Expected a PEFT model exposing save_pretrained()")
        save_pretrained(temporary / ADAPTER_DIRNAME, safe_serialization=True)

        torch.save(
            collect_extra_trainable_state(model),
            temporary / TRAINABLE_STATE_FILENAME,
        )
        payload = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "sampler": _sampler_state_dict(sampler),
            "trainer_state": dict(trainer_state),
            "rng_state": capture_rng_state(),
        }
        torch.save(payload, temporary / TRAINER_STATE_FILENAME)
        _atomic_write_json(
            temporary / METADATA_FILENAME,
            {
                "format_version": 1,
                "created_at": utc_now(),
                "trainer_state": trainer_state,
                **(dict(metadata) if metadata else {}),
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        # Leave the uniquely named temporary directory intact for diagnosis.  It
        # is never considered resumable because metadata is outside destination.
        raise
    return destination


def _torch_load(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _active_adapter_name(model: nn.Module) -> str:
    active = getattr(model, "active_adapter", "default")
    if isinstance(active, (tuple, list)):
        if len(active) != 1:
            raise ValueError(f"Expected one active adapter, got {active!r}")
        active = active[0]
    return str(active)


def _restore_adapter_weights(model: nn.Module, adapter_dir: Path, device: torch.device) -> None:
    try:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    except ImportError:
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights

    weights = load_peft_weights(str(adapter_dir), device=str(device))
    set_peft_model_state_dict(
        model,
        weights,
        adapter_name=_active_adapter_name(model),
    )


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    sampler: Any | None = None,
    device: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint into an already constructed LoRA training stack."""

    checkpoint = Path(checkpoint_dir)
    required = (
        checkpoint / ADAPTER_DIRNAME,
        checkpoint / TRAINABLE_STATE_FILENAME,
        checkpoint / TRAINER_STATE_FILENAME,
        checkpoint / METADATA_FILENAME,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint; missing: {', '.join(missing)}")

    map_location = torch.device(device)
    _restore_adapter_weights(model, checkpoint / ADAPTER_DIRNAME, map_location)
    extra_state = _torch_load(
        checkpoint / TRAINABLE_STATE_FILENAME,
        map_location=map_location,
    )
    restore_extra_trainable_state(model, extra_state)

    payload = _torch_load(checkpoint / TRAINER_STATE_FILENAME, map_location=map_location)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    sampler_state = payload.get("sampler")
    if sampler is not None and sampler_state is not None:
        method = getattr(sampler, "load_state_dict", None)
        if not callable(method):
            raise TypeError("Sampler checkpoint exists but sampler has no load_state_dict()")
        method(sampler_state)
    if restore_rng:
        restore_rng_state(payload["rng_state"])

    with (checkpoint / METADATA_FILENAME).open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return {
        "trainer_state": dict(payload.get("trainer_state", {})),
        "metadata": metadata,
        "checkpoint_dir": str(checkpoint.resolve()),
    }


__all__ = [
    "ADAPTER_DIRNAME",
    "METADATA_FILENAME",
    "ProgressReporter",
    "TRAINABLE_STATE_FILENAME",
    "TRAINER_STATE_FILENAME",
    "capture_rng_state",
    "collect_extra_trainable_state",
    "load_training_checkpoint",
    "restore_extra_trainable_state",
    "restore_rng_state",
    "save_training_checkpoint",
    "utc_now",
]
