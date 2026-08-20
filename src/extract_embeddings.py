"""Extract schema-v2 document text vectors and globally unique image vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

try:
    from document_schema import (
        SCHEMA_VERSION,
        build_retrieval_meta,
        normalize_documents,
        read_jsonl,
    )
    from model_fingerprint import (
        MetadataMismatchError,
        assert_metadata_compatible,
        build_compatibility_metadata,
        compute_data_manifest_hash,
        compute_model_fingerprint,
    )
except ImportError:
    from src.document_schema import (
        SCHEMA_VERSION,
        build_retrieval_meta,
        normalize_documents,
        read_jsonl,
    )
    from src.model_fingerprint import (
        MetadataMismatchError,
        assert_metadata_compatible,
        build_compatibility_metadata,
        compute_data_manifest_hash,
        compute_model_fingerprint,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default="data/processed/demo/documents.jsonl")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_dir", default="outputs/demo_docs_v2")
    parser.add_argument("--model_name", required=True, help="本地完整 Hugging Face 模型目录")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text_batch_size", type=int, default=256)
    parser.add_argument("--image_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flush_every_batches", type=int, default=10)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_retrieval_config(model_dir: Path) -> dict[str, Any]:
    value = load_json(model_dir / "retrieval_config.json")
    return value if isinstance(value, dict) else {}


def resolve_max_length(requested: int | None, config: dict[str, Any]) -> int:
    configured = config.get("max_length")
    if configured is not None:
        configured = int(configured)
        if requested is not None and int(requested) != configured:
            raise MetadataMismatchError("命令行 max_length 与模型 retrieval_config 不一致")
        return configured
    return 256 if requested is None else int(requested)


def projection_dim(model: ChineseCLIPModel) -> int:
    value = getattr(model.config, "projection_dim", None)
    if value is None:
        value = getattr(getattr(model.config, "text_config", None), "projection_dim", None)
    if value is None or int(value) <= 0:
        raise ValueError("模型配置缺少有效 projection_dim")
    return int(value)


def finish_features(features: Any) -> np.ndarray:
    if not isinstance(features, torch.Tensor):
        for name in ("text_embeds", "image_embeds", "pooler_output"):
            value = getattr(features, name, None)
            if isinstance(value, torch.Tensor):
                features = value
                break
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"无法从 {type(features)!r} 读取向量")
    return F.normalize(features.float(), p=2, dim=-1).detach().cpu().numpy().astype("float32")


def load_image(project_root: Path, image_path: str) -> Image.Image:
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    return Image.open(path).convert("RGB")


def open_array(path: Path, shape: tuple[int, int], resume: bool) -> np.memmap:
    if path.exists():
        if not resume:
            raise FileExistsError(f"向量文件已存在：{path}")
        array = np.load(path, mmap_mode="r+")
        if tuple(array.shape) != shape or array.dtype != np.float32:
            raise MetadataMismatchError(f"断点矩阵 {path.name} 的 shape/dtype 不一致")
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)


def prepare_state(
    output_dir: Path,
    meta: dict[str, Any],
    compatibility: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    state_path = output_dir / "extract_state.json"
    meta_path = output_dir / "retrieval_meta.json"
    existing_state = load_json(state_path)
    existing_meta = load_json(meta_path)
    if existing_state is not None or existing_meta is not None:
        if not resume or not isinstance(existing_state, dict) or not isinstance(existing_meta, dict):
            raise FileExistsError("输出目录已有不完整产物，且无法安全续跑")
        assert_metadata_compatible(existing_state, compatibility, context="断点状态")
        assert_metadata_compatible(existing_meta, compatibility, context="断点元数据")
        if existing_meta.get("retrieval_manifest_hash") != meta["retrieval_manifest_hash"]:
            raise MetadataMismatchError("documents.jsonl 与已有 retrieval_meta 不一致")
        return existing_state
    save_json(meta, meta_path)
    state = {
        **compatibility,
        "schema_version": SCHEMA_VERSION,
        "retrieval_manifest_hash": meta["retrieval_manifest_hash"],
        "document_count": meta["document_count"],
        "image_count": meta["image_count"],
        "relation_count": meta["relation_count"],
        "text_documents_done": 0,
        "image_assets_done": 0,
        "complete": False,
    }
    save_json(state, state_path)
    return state


def encode_texts(
    documents: list[dict[str, Any]],
    output: np.memmap,
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    start = int(state.get("text_documents_done", 0))
    for batch_number, batch_start in enumerate(
        tqdm(range(start, len(documents), args.text_batch_size), desc="文档文本向量"), start=1
    ):
        end = min(batch_start + args.text_batch_size, len(documents))
        encoded = processor(
            text=[document["text"] for document in documents[batch_start:end]],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.fp16
        ):
            output[batch_start:end] = finish_features(model.get_text_features(**encoded))
        state["text_documents_done"] = end
        if batch_number % args.flush_every_batches == 0 or end == len(documents):
            output.flush()
            save_json(state, state_path)


def encode_images(
    image_assets: list[dict[str, Any]],
    output: np.memmap,
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    project_root: Path,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    start = int(state.get("image_assets_done", 0))
    for batch_number, batch_start in enumerate(
        tqdm(range(start, len(image_assets), args.image_batch_size), desc="唯一图片向量"), start=1
    ):
        end = min(batch_start + args.image_batch_size, len(image_assets))
        images = [load_image(project_root, asset["image_path"]) for asset in image_assets[batch_start:end]]
        encoded = processor(images=images, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.fp16
        ):
            output[batch_start:end] = finish_features(model.get_image_features(**encoded))
        state["image_assets_done"] = end
        if batch_number % args.flush_every_batches == 0 or end == len(image_assets):
            output.flush()
            save_json(state, state_path)


def main() -> int:
    args = parse_args()
    if args.text_batch_size <= 0 or args.image_batch_size <= 0 or args.flush_every_batches <= 0:
        raise ValueError("batch size 和 flush_every_batches 必须为正整数")
    model_dir = Path(args.model_name)
    if not model_dir.is_dir():
        raise FileNotFoundError("--model_name 必须指向本地完整模型目录，以便计算稳定指纹")
    raw_documents = read_jsonl(args.documents)
    if args.limit is not None:
        raw_documents = raw_documents[: args.limit]
    documents, image_assets = normalize_documents(raw_documents)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(args.project_root)
    config = load_retrieval_config(model_dir)
    args.max_length = resolve_max_length(args.max_length, config)
    device = choose_device(args.device)
    processor = ChineseCLIPProcessor.from_pretrained(model_dir)
    model = ChineseCLIPModel.from_pretrained(model_dir).eval().to(device)
    dimension = projection_dim(model)
    if config.get("dimension") is not None and int(config["dimension"]) != dimension:
        raise MetadataMismatchError("模型 retrieval_config.dimension 与实际模型不一致")
    compatibility = build_compatibility_metadata(
        model_fingerprint=compute_model_fingerprint(model_dir),
        data_manifest_hash=compute_data_manifest_hash(Path(args.documents)),
        dimension=dimension,
        normalization="l2",
        max_length=args.max_length,
    )
    meta = build_retrieval_meta(
        documents, image_assets, compatibility, source_items_path=str(args.documents)
    )
    state = prepare_state(output_dir, meta, compatibility, args.resume)
    text_array = open_array(
        output_dir / "text_embeddings.npy", (len(documents), dimension), args.resume
    )
    image_array = open_array(
        output_dir / "image_embeddings.npy", (len(image_assets), dimension), args.resume
    )
    state_path = output_dir / "extract_state.json"
    encode_texts(documents, text_array, processor, model, device, args, state, state_path)
    encode_images(
        image_assets, image_array, processor, model, device, project_root, args, state, state_path
    )
    state["complete"] = True
    save_json(state, state_path)
    print(
        json.dumps(
            {"document_count": len(documents), "image_count": len(image_assets), "dimension": dimension},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
