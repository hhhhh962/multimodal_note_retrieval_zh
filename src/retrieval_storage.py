"""Disk-backed metadata stores for large schema-v2 retrieval artifacts."""

from __future__ import annotations

import hashlib
import json
import mmap
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_with_offsets(
    records: Sequence[Mapping[str, Any]], jsonl_path: Path, offsets_path: Path
) -> None:
    offsets = np.empty(len(records) + 1, dtype=np.uint64)
    position = 0
    with jsonl_path.open("wb") as handle:
        for row, record in enumerate(records):
            offsets[row] = position
            payload = json.dumps(
                record, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8") + b"\n"
            handle.write(payload)
            position += len(payload)
    offsets[len(records)] = position
    np.save(offsets_path, offsets)


def _write_relations(
    documents: Sequence[Mapping[str, Any]],
    image_assets: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    image_doc_offsets = np.empty(len(image_assets) + 1, dtype=np.uint64)
    image_doc_rows: list[int] = []
    image_doc_offsets[0] = 0
    doc_counts = np.zeros(len(documents), dtype=np.uint32)
    for image_row, asset in enumerate(image_assets):
        owners = [int(value) for value in asset["doc_row_ids"]]
        image_doc_rows.extend(owners)
        for owner in owners:
            doc_counts[owner] += 1
        image_doc_offsets[image_row + 1] = len(image_doc_rows)

    doc_image_offsets = np.empty(len(documents) + 1, dtype=np.uint64)
    doc_image_offsets[0] = 0
    np.cumsum(doc_counts, dtype=np.uint64, out=doc_image_offsets[1:])
    doc_image_rows = np.empty(len(image_doc_rows), dtype=np.uint32)
    cursors = doc_image_offsets[:-1].copy()
    for image_row, asset in enumerate(image_assets):
        for raw_owner in asset["doc_row_ids"]:
            owner = int(raw_owner)
            position = int(cursors[owner])
            doc_image_rows[position] = image_row
            cursors[owner] += 1

    np.save(output_dir / "image_doc_offsets.npy", image_doc_offsets)
    np.save(output_dir / "image_doc_rows.npy", np.asarray(image_doc_rows, dtype=np.uint32))
    np.save(output_dir / "doc_image_offsets.npy", doc_image_offsets)
    np.save(output_dir / "doc_image_rows.npy", doc_image_rows)


def write_external_metadata(
    meta: Mapping[str, Any],
    documents: Sequence[Mapping[str, Any]],
    image_assets: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write JSONL/offset/CSR artifacts and return a compact metadata header."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    documents_path = root / "documents.jsonl"
    document_offsets_path = root / "document_offsets.npy"
    image_assets_path = root / "image_assets.jsonl"
    image_offsets_path = root / "image_asset_offsets.npy"
    _write_jsonl_with_offsets(documents, documents_path, document_offsets_path)
    _write_jsonl_with_offsets(image_assets, image_assets_path, image_offsets_path)
    _write_relations(documents, image_assets, root)

    compact = {key: value for key, value in meta.items() if key not in {"documents", "image_assets"}}
    compact["metadata_storage"] = "jsonl_offsets_csr_v1"
    compact["documents"] = {
        "count": len(documents),
        "file": documents_path.name,
        "offsets_file": document_offsets_path.name,
    }
    compact["image_assets"] = {
        "count": len(image_assets),
        "file": image_assets_path.name,
        "offsets_file": image_offsets_path.name,
    }
    compact["relations"] = {
        "count": int(meta["relation_count"]),
        "doc_image_offsets_file": "doc_image_offsets.npy",
        "doc_image_rows_file": "doc_image_rows.npy",
        "image_doc_offsets_file": "image_doc_offsets.npy",
        "image_doc_rows_file": "image_doc_rows.npy",
    }
    artifact_names = [
        documents_path.name,
        document_offsets_path.name,
        image_assets_path.name,
        image_offsets_path.name,
        "doc_image_offsets.npy",
        "doc_image_rows.npy",
        "image_doc_offsets.npy",
        "image_doc_rows.npy",
    ]
    compact["metadata_artifact_sha256"] = {
        name: sha256_file(root / name) for name in artifact_names
    }
    return compact


class JsonlOffsetStore(Sequence[dict[str, Any]]):
    """Random-access JSONL sequence without materializing every record."""

    def __init__(self, jsonl_path: Path, offsets_path: Path, expected_count: int) -> None:
        self.path = jsonl_path
        self.offsets = np.load(offsets_path)
        if self.offsets.dtype != np.uint64 or tuple(self.offsets.shape) != (expected_count + 1,):
            raise ValueError(f"偏移表不匹配：{offsets_path}")
        self._handle = jsonl_path.open("rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        if int(self.offsets[-1]) != len(self._mmap):
            raise ValueError(f"偏移表末尾与 JSONL 大小不一致：{jsonl_path}")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        row = int(index)
        if row < 0:
            row += len(self)
        if row < 0 or row >= len(self):
            raise IndexError(row)
        start, end = int(self.offsets[row]), int(self.offsets[row + 1])
        return json.loads(self._mmap[start:end])

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
        if getattr(self, "_handle", None) is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class CsrRows(Sequence[np.ndarray]):
    def __init__(self, offsets_path: Path, rows_path: Path, expected_count: int) -> None:
        self.offsets = np.load(offsets_path)
        self.rows = np.load(rows_path)
        if tuple(self.offsets.shape) != (expected_count + 1,):
            raise ValueError(f"CSR offsets 数量不一致：{offsets_path}")
        if int(self.offsets[-1]) != len(self.rows):
            raise ValueError(f"CSR offsets 与 rows 长度不一致：{rows_path}")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        row = int(index)
        if row < 0:
            row += len(self)
        if row < 0 or row >= len(self):
            raise IndexError(row)
        return self.rows[int(self.offsets[row]) : int(self.offsets[row + 1])]


def load_external_metadata(meta: Mapping[str, Any], root: str | Path):
    base = Path(root)
    documents = meta["documents"]
    images = meta["image_assets"]
    relations = meta["relations"]
    document_store = JsonlOffsetStore(
        base / documents["file"], base / documents["offsets_file"], int(documents["count"])
    )
    image_store = JsonlOffsetStore(
        base / images["file"], base / images["offsets_file"], int(images["count"])
    )
    doc_image_rows = CsrRows(
        base / relations["doc_image_offsets_file"],
        base / relations["doc_image_rows_file"],
        int(documents["count"]),
    )
    return document_store, image_store, doc_image_rows


def external_relation_arrays(meta: Mapping[str, Any], root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    base = Path(root)
    relations = meta["relations"]
    image_offsets = np.load(base / relations["image_doc_offsets_file"], mmap_mode="r")
    doc_rows = np.load(base / relations["image_doc_rows_file"], mmap_mode="r")
    counts = np.diff(image_offsets).astype(np.int64, copy=False)
    image_rows = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return image_rows, np.asarray(doc_rows, dtype=np.int64)
