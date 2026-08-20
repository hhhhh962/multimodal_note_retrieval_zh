"""Document-level retrieval schema shared by extraction, migration, and search."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2


def make_doc_id(item: Mapping[str, Any], row_id: int | None = None) -> str:
    """Build the stable document identifier used by every retrieval data source."""

    source = str(item.get("source") or "unknown")
    split = str(item.get("split") or "unknown")
    text_id = item.get("text_id")
    if text_id is not None and str(text_id) != "":
        identity = str(text_id)
    else:
        fallback = item.get("item_id") or item.get("note_id")
        if fallback is None:
            fallback = row_id if row_id is not None else "unknown"
        identity = str(fallback)
    return f"{source}:{split}:{identity}"


def item_text(item: Mapping[str, Any]) -> str:
    """Read text from either the normalized or legacy note representation."""

    text = item.get("text")
    if text is not None:
        return str(text).strip()
    parts = [item.get("title"), item.get("content")]
    return " ".join(str(part).strip() for part in parts if part).strip()


def _image_from_pair(item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = str(item.get("image_path") or "").strip()
    if not path:
        return None
    return {"image_id": item.get("image_id"), "image_path": path}


def _normalized_images(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_images = document.get("images")
    if not isinstance(raw_images, Sequence) or isinstance(raw_images, (str, bytes)):
        raise ValueError(f"document {document.get('doc_id')!r} 的 images 必须是列表")
    images: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw in raw_images:
        if not isinstance(raw, Mapping):
            raise ValueError(f"document {document.get('doc_id')!r} 包含非对象图片项")
        path = str(raw.get("image_path") or "").strip()
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        images.append({"image_id": raw.get("image_id"), "image_path": path})
    if not images:
        raise ValueError(f"document {document.get('doc_id')!r} 没有有效图片")
    return images


def build_documents_from_pairs(
    items: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int], list[int]]:
    """Group legacy pair rows into documents and globally unique image assets.

    Returns ``documents``, ``image_assets``, ``pair_doc_rows`` and
    ``pair_image_rows``. Shared images are encoded once and list every owning
    document row.
    """

    documents: list[dict[str, Any]] = []
    image_assets: list[dict[str, Any]] = []
    doc_rows: dict[str, int] = {}
    image_rows: dict[str, int] = {}
    document_image_paths: list[set[str]] = []
    image_owner_sets: list[set[int]] = []
    pair_doc_rows: list[int] = []
    pair_image_rows: list[int] = []

    for pair_row, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"pair row {pair_row} 不是对象")
        doc_id = make_doc_id(item, pair_row)
        text = item_text(item)
        if not text:
            raise ValueError(f"document {doc_id!r} 的文本为空")
        doc_row = doc_rows.get(doc_id)
        if doc_row is None:
            doc_row = len(documents)
            doc_rows[doc_id] = doc_row
            documents.append(
                {
                    "doc_row_id": doc_row,
                    "doc_id": doc_id,
                    "text": text,
                    "source": item.get("source"),
                    "split": item.get("split"),
                    "text_id": item.get("text_id"),
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "tags": item.get("tags"),
                    "item_ids": [],
                    "images": [],
                }
            )
            document_image_paths.append(set())
        elif documents[doc_row]["text"] != text:
            raise ValueError(
                f"document {doc_id!r} 出现冲突文本："
                f"{documents[doc_row]['text']!r} != {text!r}"
            )

        item_id = item.get("item_id") or item.get("note_id")
        if item_id is not None and item_id not in documents[doc_row]["item_ids"]:
            documents[doc_row]["item_ids"].append(item_id)

        image = _image_from_pair(item)
        if image is None:
            raise ValueError(f"pair row {pair_row} 缺少 image_path")
        path = image["image_path"]
        if path not in document_image_paths[doc_row]:
            document_image_paths[doc_row].add(path)
            documents[doc_row]["images"].append(image)

        image_row = image_rows.get(path)
        if image_row is None:
            image_row = len(image_assets)
            image_rows[path] = image_row
            image_assets.append(
                {
                    "image_row_id": image_row,
                    "image_id": image.get("image_id"),
                    "image_path": path,
                    "doc_row_ids": [],
                }
            )
            image_owner_sets.append(set())
        if doc_row not in image_owner_sets[image_row]:
            image_owner_sets[image_row].add(doc_row)
            image_assets[image_row]["doc_row_ids"].append(doc_row)

        pair_doc_rows.append(doc_row)
        pair_image_rows.append(image_row)

    if not documents:
        raise ValueError("没有可用文档")
    return documents, image_assets, pair_doc_rows, pair_image_rows


def normalize_documents(
    raw_documents: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate v2 document input and build globally unique image assets."""

    documents: list[dict[str, Any]] = []
    image_assets: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    image_rows: dict[str, int] = {}
    image_owner_sets: list[set[int]] = []

    for row, raw in enumerate(raw_documents):
        if not isinstance(raw, Mapping):
            raise ValueError(f"document row {row} 不是对象")
        doc_id = str(raw.get("doc_id") or make_doc_id(raw, row))
        if doc_id in seen_doc_ids:
            raise ValueError(f"重复 doc_id：{doc_id}")
        seen_doc_ids.add(doc_id)
        text = item_text(raw)
        if not text:
            raise ValueError(f"document {doc_id!r} 的文本为空")
        images = _normalized_images(raw)
        doc_row = len(documents)
        document = {
            "doc_row_id": doc_row,
            "doc_id": doc_id,
            "text": text,
            "source": raw.get("source"),
            "split": raw.get("split"),
            "text_id": raw.get("text_id"),
            "title": raw.get("title"),
            "content": raw.get("content"),
            "tags": raw.get("tags"),
            "item_ids": list(raw.get("item_ids") or []),
            "images": images,
        }
        documents.append(document)
        for image in images:
            path = image["image_path"]
            image_row = image_rows.get(path)
            if image_row is None:
                image_row = len(image_assets)
                image_rows[path] = image_row
                image_assets.append(
                    {
                        "image_row_id": image_row,
                        "image_id": image.get("image_id"),
                        "image_path": path,
                        "doc_row_ids": [],
                    }
                )
                image_owner_sets.append(set())
            if doc_row not in image_owner_sets[image_row]:
                image_owner_sets[image_row].add(doc_row)
                image_assets[image_row]["doc_row_ids"].append(doc_row)
    if not documents:
        raise ValueError("没有可用文档")
    return documents, image_assets


def retrieval_manifest_hash(
    documents: Sequence[Mapping[str, Any]], image_assets: Sequence[Mapping[str, Any]]
) -> str:
    """Hash the normalized v2 retrieval relation independently of file paths."""

    digest = hashlib.sha256()
    for collection in (documents, image_assets):
        for value in collection:
            digest.update(
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest()


def build_retrieval_meta(
    documents: list[dict[str, Any]],
    image_assets: list[dict[str, Any]],
    compatibility: Mapping[str, Any],
    *,
    source_items_path: str | None = None,
) -> dict[str, Any]:
    relations = sum(len(asset["doc_row_ids"]) for asset in image_assets)
    return {
        **dict(compatibility),
        "schema_version": SCHEMA_VERSION,
        "document_count": len(documents),
        "image_count": len(image_assets),
        "relation_count": relations,
        "retrieval_manifest_hash": retrieval_manifest_hash(documents, image_assets),
        "source_items_path": source_items_path,
        "documents": documents,
        "image_assets": image_assets,
    }


def validate_retrieval_meta(
    meta: Mapping[str, Any], artifact_dir: str | Path | None = None
) -> None:
    """Fail fast on pair-level or internally inconsistent metadata."""

    if int(meta.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            "只支持文档级检索 schema_version=2；检测到旧 pair-level 产物，"
            "请运行 scripts/migrate_pair_artifacts.py"
        )
    documents = meta.get("documents")
    image_assets = meta.get("image_assets")
    if isinstance(documents, Mapping) and isinstance(image_assets, Mapping):
        if meta.get("metadata_storage") != "jsonl_offsets_csr_v1":
            raise ValueError("外置元数据缺少 metadata_storage=jsonl_offsets_csr_v1")
        if int(documents.get("count", -1)) != int(meta.get("document_count", -2)):
            raise ValueError("documents.count 与 document_count 不一致")
        if int(image_assets.get("count", -1)) != int(meta.get("image_count", -2)):
            raise ValueError("image_assets.count 与 image_count 不一致")
        relations = meta.get("relations")
        if not isinstance(relations, Mapping) or int(relations.get("count", -1)) != int(
            meta.get("relation_count", -2)
        ):
            raise ValueError("relations.count 与 relation_count 不一致")
        if artifact_dir is not None:
            _validate_external_artifacts(meta, Path(artifact_dir))
        return
    if not isinstance(documents, list) or not isinstance(image_assets, list):
        raise ValueError("retrieval_meta 缺少 documents 或 image_assets")
    if int(meta.get("document_count", -1)) != len(documents):
        raise ValueError("retrieval_meta.document_count 与 documents 数量不一致")
    if int(meta.get("image_count", -1)) != len(image_assets):
        raise ValueError("retrieval_meta.image_count 与 image_assets 数量不一致")
    document_paths: list[set[str]] = []
    for row, document in enumerate(documents):
        if int(document.get("doc_row_id", -1)) != row:
            raise ValueError(f"document row {row} 的 doc_row_id 不连续")
        normalized = _normalized_images(document)
        document_paths.append({str(image["image_path"]) for image in normalized})
    seen_image_paths: set[str] = set()
    relation_count = 0
    for row, image in enumerate(image_assets):
        if int(image.get("image_row_id", -1)) != row:
            raise ValueError(f"image row {row} 的 image_row_id 不连续")
        path = str(image.get("image_path") or "").strip()
        if not path or path in seen_image_paths:
            raise ValueError(f"image row {row} 的 image_path 为空或重复")
        seen_image_paths.add(path)
        owners = image.get("doc_row_ids")
        if not isinstance(owners, list) or not owners:
            raise ValueError(f"image row {row} 没有所属文档")
        normalized_owners = [int(owner) for owner in owners]
        if len(normalized_owners) != len(set(normalized_owners)):
            raise ValueError(f"image row {row} 包含重复 doc_row_id")
        for owner in normalized_owners:
            if owner < 0 or owner >= len(documents):
                raise ValueError(f"image row {row} 包含越界 doc_row_id")
            if path not in document_paths[owner]:
                raise ValueError(f"image row {row} 与 document row {owner} 的图片关系不一致")
        relation_count += len(normalized_owners)
    if int(meta.get("relation_count", -1)) != relation_count:
        raise ValueError("retrieval_meta.relation_count 与图片关系数量不一致")
    if meta.get("retrieval_manifest_hash") != retrieval_manifest_hash(documents, image_assets):
        raise ValueError("retrieval_meta.retrieval_manifest_hash 与实际内容不一致")


def _validate_external_artifacts(meta: Mapping[str, Any], root: Path) -> None:
    """Validate disk-backed JSONL/offset/CSR metadata without loading all records."""

    import numpy as np

    try:
        from retrieval_storage import JsonlOffsetStore, CsrRows, sha256_file
    except ImportError:
        from src.retrieval_storage import JsonlOffsetStore, CsrRows, sha256_file

    documents = meta["documents"]
    images = meta["image_assets"]
    relations = meta["relations"]
    checksums = meta.get("metadata_artifact_sha256")
    if not isinstance(checksums, Mapping) or not checksums:
        raise ValueError("metadata_artifact_sha256 缺失")
    for name, expected in checksums.items():
        path = root / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"元数据产物缺失：{path}")
        if sha256_file(path) != expected:
            raise ValueError(f"元数据产物 SHA256 不一致：{path.name}")

    document_store = JsonlOffsetStore(
        root / documents["file"],
        root / documents["offsets_file"],
        int(documents["count"]),
    )
    image_store = JsonlOffsetStore(
        root / images["file"],
        root / images["offsets_file"],
        int(images["count"]),
    )
    try:
        if len(document_store) != int(meta["document_count"]):
            raise ValueError("documents JSONL 数量不一致")
        if len(image_store) != int(meta["image_count"]):
            raise ValueError("image_assets JSONL 数量不一致")
    finally:
        document_store.close()
        image_store.close()

    doc_image = CsrRows(
        root / relations["doc_image_offsets_file"],
        root / relations["doc_image_rows_file"],
        int(meta["document_count"]),
    )
    image_doc = CsrRows(
        root / relations["image_doc_offsets_file"],
        root / relations["image_doc_rows_file"],
        int(meta["image_count"]),
    )
    if len(doc_image.rows) != int(meta["relation_count"]) or len(image_doc.rows) != int(
        meta["relation_count"]
    ):
        raise ValueError("CSR relation_count 不一致")
    if len(doc_image.rows) and int(np.max(doc_image.rows)) >= int(meta["image_count"]):
        raise ValueError("doc_image_rows 包含越界图片行")
    if len(image_doc.rows) and int(np.max(image_doc.rows)) >= int(meta["document_count"]):
        raise ValueError("image_doc_rows 包含越界文档行")
    if np.any(np.diff(doc_image.offsets) == 0):
        raise ValueError("存在没有图片的文档")
    if np.any(np.diff(image_doc.offsets) == 0):
        raise ValueError("存在没有所属文档的图片")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def write_documents_jsonl(documents: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
