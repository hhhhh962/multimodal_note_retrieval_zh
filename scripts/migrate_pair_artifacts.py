#!/usr/bin/env python3
"""Migrate aligned pair-level embeddings to document-level retrieval schema v2."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_schema import (  # noqa: E402
    SCHEMA_VERSION,
    build_documents_from_pairs,
    build_retrieval_meta,
    validate_retrieval_meta,
)
from src.model_fingerprint import REQUIRED_COMPATIBILITY_FIELDS  # noqa: E402
from src.retrieval_storage import write_external_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--items", type=Path, default=None, help="旧 pair-level items.jsonl；全量迁移建议显式传入")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="重复向量逐元素绝对误差上限；默认严格使用 1e-6，放宽时会在报告中记录实际漂移",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def compatibility_from_source(meta: Mapping[str, Any] | None, state: Mapping[str, Any]) -> dict[str, Any]:
    compatibility: dict[str, Any] = {}
    for field in REQUIRED_COMPATIBILITY_FIELDS:
        left = None if meta is None else meta.get(field)
        right = state.get(field)
        if right is None:
            raise ValueError(f"旧 extract_state 的 {field} 缺失")
        if left is not None and left != right:
            raise ValueError(f"旧 item_meta 与 extract_state 的 {field} 不一致")
        compatibility[field] = right
    compatibility["source_data_manifest_hash"] = compatibility["data_manifest_hash"]
    if meta is not None and "max_length" in meta:
        compatibility["max_length"] = meta["max_length"]
    return compatibility


def _validate_source_arrays(
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    count: int,
    dimension: int,
) -> None:
    expected = (count, dimension)
    if tuple(text_embeddings.shape) != expected:
        raise ValueError(f"旧 text_embeddings shape={text_embeddings.shape}，期望 {expected}")
    if tuple(image_embeddings.shape) != expected:
        raise ValueError(f"旧 image_embeddings shape={image_embeddings.shape}，期望 {expected}")


def normalized_vector(vector: np.ndarray, pair_row: int, label: str) -> tuple[np.ndarray, float]:
    if not np.isfinite(vector).all():
        raise ValueError(f"旧{label}向量包含 NaN/Inf：pair_row={pair_row}")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"旧{label}向量范数无效：pair_row={pair_row} norm={norm}")
    return np.asarray(vector / norm, dtype="float32"), norm


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            yield value


def migrate(
    source_dir: Path,
    output_dir: Path,
    atol: float = 1e-6,
    items_path: Path | None = None,
) -> dict[str, Any]:
    """Perform an atomic, non-overwriting migration and return its report."""

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if atol < 0:
        raise ValueError("atol 不能为负数")
    if output_dir.exists():
        raise FileExistsError(f"目标目录已存在，拒绝覆盖：{output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"发现遗留迁移临时目录：{temporary}")

    item_meta_path = source_dir / "item_meta.json"
    state_path = source_dir / "extract_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("旧产物缺少 extract_state.json")
    state = load_json(state_path)
    item_meta = None
    if items_path is None:
        discovered = state.get("items") or state.get("items_path")
        if discovered and Path(discovered).is_file():
            items_path = Path(discovered)
        elif item_meta_path.is_file():
            item_meta = load_json(item_meta_path)
            discovered = item_meta.get("items_path") if isinstance(item_meta, dict) else None
            if discovered and Path(discovered).is_file():
                items_path = Path(discovered)
    if int((item_meta or {}).get("schema_version", 1)) == SCHEMA_VERSION:
        raise ValueError("源目录已经是 schema v2，无需迁移")
    fallback_items = (item_meta or {}).get("items")
    if items_path is not None:
        pair_source = iter_jsonl(Path(items_path))
    elif isinstance(fallback_items, list) and fallback_items:
        pair_source = iter(fallback_items)
    else:
        raise ValueError("无法找到旧 pair 清单；请显式传 --items /path/to/items.jsonl")
    count = int(state.get("count", (item_meta or {}).get("count", -1)))
    dimension = int(state.get("dimension", (item_meta or {}).get("dimension", 0)))
    if count <= 0 or dimension <= 0:
        raise ValueError("旧 extract_state 的 count 或 dimension 无效")
    compatibility = compatibility_from_source(item_meta, state)

    source_text = np.load(source_dir / "text_embeddings.npy", mmap_mode="r")
    source_image = np.load(source_dir / "image_embeddings.npy", mmap_mode="r")
    _validate_source_arrays(source_text, source_image, count, dimension)

    documents, image_assets, pair_doc_rows, pair_image_rows = build_documents_from_pairs(pair_source)
    if len(pair_doc_rows) != count:
        raise ValueError(f"pair 清单行数={len(pair_doc_rows)}，旧向量 count={count}")
    meta = build_retrieval_meta(
        documents,
        image_assets,
        compatibility,
        source_items_path=str(items_path or (item_meta or {}).get("items_path") or item_meta_path),
    )
    validate_retrieval_meta(meta)

    temporary.mkdir(parents=True)
    target_text = None
    target_image = None
    try:
        target_text = np.lib.format.open_memmap(
            temporary / "text_embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(len(documents), dimension),
        )
        target_image = np.lib.format.open_memmap(
            temporary / "image_embeddings.npy",
            mode="w+",
            dtype="float32",
            shape=(len(image_assets), dimension),
        )
        seen_docs = np.zeros(len(documents), dtype=np.bool_)
        seen_images = np.zeros(len(image_assets), dtype=np.bool_)
        duplicate_text_rows = 0
        duplicate_image_rows = 0
        nonidentical_text_rows = 0
        nonidentical_image_rows = 0
        max_duplicate_text_abs_diff = 0.0
        max_duplicate_image_abs_diff = 0.0
        source_text_norm_min = float("inf")
        source_text_norm_max = 0.0
        source_image_norm_min = float("inf")
        source_image_norm_max = 0.0

        for pair_row, (doc_row, image_row) in enumerate(zip(pair_doc_rows, pair_image_rows)):
            text_vector, text_norm = normalized_vector(
                np.asarray(source_text[pair_row], dtype="float32"), pair_row, "文本"
            )
            image_vector, image_norm = normalized_vector(
                np.asarray(source_image[pair_row], dtype="float32"), pair_row, "图片"
            )
            source_text_norm_min = min(source_text_norm_min, text_norm)
            source_text_norm_max = max(source_text_norm_max, text_norm)
            source_image_norm_min = min(source_image_norm_min, image_norm)
            source_image_norm_max = max(source_image_norm_max, image_norm)
            if not seen_docs[doc_row]:
                target_text[doc_row] = text_vector
                seen_docs[doc_row] = True
            else:
                duplicate_text_rows += 1
                max_diff = float(np.max(np.abs(target_text[doc_row] - text_vector)))
                max_duplicate_text_abs_diff = max(max_duplicate_text_abs_diff, max_diff)
                if max_diff > 0.0:
                    nonidentical_text_rows += 1
                if max_diff > atol:
                    raise ValueError(
                        f"同一文档的重复文本向量不一致：pair_row={pair_row} "
                        f"doc_row={doc_row} max_diff={max_diff} atol={atol}"
                    )

            if not seen_images[image_row]:
                target_image[image_row] = image_vector
                seen_images[image_row] = True
            else:
                duplicate_image_rows += 1
                max_diff = float(np.max(np.abs(target_image[image_row] - image_vector)))
                max_duplicate_image_abs_diff = max(max_duplicate_image_abs_diff, max_diff)
                if max_diff > 0.0:
                    nonidentical_image_rows += 1
                if max_diff > atol:
                    raise ValueError(
                        f"同一路径的重复图片向量不一致：pair_row={pair_row} "
                        f"image_row={image_row} max_diff={max_diff} atol={atol}"
                    )

        if not bool(seen_docs.all()) or not bool(seen_images.all()):
            raise RuntimeError("迁移结束后仍有文档或图片向量未写入")
        target_text.flush()
        target_image.flush()
        del target_text, target_image
        target_text = None
        target_image = None

        compact_meta = write_external_metadata(meta, documents, image_assets, temporary)
        validate_retrieval_meta(compact_meta, temporary)
        write_json(temporary / "retrieval_meta.json", compact_meta)
        extraction_state = {
            **compatibility,
            "schema_version": SCHEMA_VERSION,
            "document_count": len(documents),
            "image_count": len(image_assets),
            "relation_count": meta["relation_count"],
            "retrieval_manifest_hash": compact_meta["retrieval_manifest_hash"],
            "text_documents_done": len(documents),
            "image_assets_done": len(image_assets),
            "complete": True,
            "migration_source": str(source_dir),
        }
        write_json(temporary / "extract_state.json", extraction_state)
        report = {
            "schema_version": SCHEMA_VERSION,
            "source_pair_count": count,
            "document_count": len(documents),
            "image_count": len(image_assets),
            "relation_count": meta["relation_count"],
            "duplicate_text_rows_validated": duplicate_text_rows,
            "duplicate_image_rows_validated": duplicate_image_rows,
            "nonidentical_text_rows_within_atol": nonidentical_text_rows,
            "nonidentical_image_rows_within_atol": nonidentical_image_rows,
            "max_duplicate_text_abs_diff": max_duplicate_text_abs_diff,
            "max_duplicate_image_abs_diff": max_duplicate_image_abs_diff,
            "source_text_norm_minmax": [source_text_norm_min, source_text_norm_max],
            "source_image_norm_minmax": [source_image_norm_min, source_image_norm_max],
            "output_l2_renormalized": True,
            "atol": atol,
        }
        write_json(temporary / "migration_report.json", report)
        os.replace(temporary, output_dir)
        return report
    except Exception:
        del target_text, target_image
        gc.collect()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    report = migrate(args.source_dir, args.output_dir, args.atol, args.items)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
