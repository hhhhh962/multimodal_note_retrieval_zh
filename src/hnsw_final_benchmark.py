"""Benchmark document-level v2 HNSW candidates against exact Flat scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss
import numpy as np

try:
    from document_schema import validate_retrieval_meta
    from document_retrieval import adaptive_image_candidates, build_doc_image_rows, exact_score_documents
    from retrieval_storage import external_relation_arrays, load_external_metadata
except ImportError:
    from src.document_schema import validate_retrieval_meta
    from src.document_retrieval import adaptive_image_candidates, build_doc_image_rows, exact_score_documents
    from src.retrieval_storage import external_relation_arrays, load_external_metadata


FIXED_EF_SEARCH = 512
FINAL_KS = (5, 10)
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_CANDIDATE_POOL_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_dir", default="outputs/finetuned_docs_v2")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate_pool_size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    return parser.parse_args()


def recall_at_k(exact: Sequence[int], approximate: Sequence[int], k: int) -> float:
    if len(exact) < k or len(approximate) < k:
        raise ValueError(f"最终结果不足 Top-{k}：exact={len(exact)}, approximate={len(approximate)}")
    return len(set(exact[:k]) & set(approximate[:k])) / float(k)


def relation_arrays(image_assets: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    image_rows: list[int] = []
    doc_rows: list[int] = []
    for image_row, asset in enumerate(image_assets):
        for doc_row in asset["doc_row_ids"]:
            image_rows.append(image_row)
            doc_rows.append(int(doc_row))
    return np.asarray(image_rows, dtype=np.int64), np.asarray(doc_rows, dtype=np.int64)


def exact_all_document_scores(
    *,
    text_query: np.ndarray | None,
    image_query: np.ndarray | None,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    relation_image_rows: np.ndarray,
    relation_doc_rows: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    has_text = text_query is not None
    has_image = image_query is not None
    if not has_text and not has_image:
        raise ValueError("至少需要一种查询")
    document_count = int(text_embeddings.shape[0])
    denominator = int(has_text) + int(has_image)

    def image_doc_max(query: np.ndarray) -> np.ndarray:
        image_scores = np.asarray(image_embeddings @ np.asarray(query).reshape(-1), dtype="float32")
        doc_scores = np.full(document_count, -np.inf, dtype="float32")
        np.maximum.at(doc_scores, relation_doc_rows, image_scores[relation_image_rows])
        return doc_scores

    text_index_scores = np.zeros(document_count, dtype="float32")
    image_index_scores = np.zeros(document_count, dtype="float32")
    if has_text:
        text_index_scores += np.asarray(text_embeddings @ np.asarray(text_query).reshape(-1), dtype="float32")
        image_index_scores += image_doc_max(np.asarray(text_query))
    if has_image:
        text_index_scores += np.asarray(text_embeddings @ np.asarray(image_query).reshape(-1), dtype="float32")
        image_index_scores += image_doc_max(np.asarray(image_query))
    text_index_scores /= denominator
    image_index_scores /= denominator
    return float(alpha) * text_index_scores + (1.0 - float(alpha)) * image_index_scores


def top_rows(scores: np.ndarray, k: int) -> list[int]:
    k = min(int(k), len(scores))
    if k <= 0:
        return []
    selected = np.argpartition(-scores, k - 1)[:k]
    return sorted((int(row) for row in selected), key=lambda row: (-float(scores[row]), row))


def hnsw_candidates(
    text_index: faiss.Index,
    image_index: faiss.Index,
    image_assets: Sequence[Mapping[str, Any]],
    text_query: np.ndarray | None,
    image_query: np.ndarray | None,
    target: int,
) -> set[int]:
    candidates: set[int] = set()
    for query in (text_query, image_query):
        if query is None:
            continue
        _, ids = text_index.search(np.asarray(query, dtype="float32").reshape(1, -1), target)
        candidates.update(int(row) for row in ids[0] if int(row) >= 0)
        image_docs, _ = adaptive_image_candidates(image_index, query, image_assets, target)
        candidates.update(image_docs)
    return candidates


def benchmark_document_recall(
    embedding_dir: Path,
    index_dir: Path,
    sample_size: int,
    candidate_pool_size: int,
) -> dict[str, Any]:
    meta = json.loads((embedding_dir / "retrieval_meta.json").read_text(encoding="utf-8"))
    validate_retrieval_meta(meta, embedding_dir)
    if meta.get("metadata_storage") == "jsonl_offsets_csr_v1":
        documents, image_assets, doc_image_rows = load_external_metadata(meta, embedding_dir)
        rel_images, rel_docs = external_relation_arrays(meta, embedding_dir)
    else:
        documents = meta["documents"]
        image_assets = meta["image_assets"]
        doc_image_rows = build_doc_image_rows(len(documents), image_assets)
        rel_images, rel_docs = relation_arrays(image_assets)
    text_embeddings = np.load(embedding_dir / "text_embeddings.npy", mmap_mode="r")
    image_embeddings = np.load(embedding_dir / "image_embeddings.npy", mmap_mode="r")
    text_index = faiss.read_index(str(index_dir / "text.index"))
    image_index = faiss.read_index(str(index_dir / "image.index"))
    text_index.hnsw.efSearch = FIXED_EF_SEARCH
    image_index.hnsw.efSearch = FIXED_EF_SEARCH
    if sample_size <= 0 or candidate_pool_size < max(FINAL_KS):
        raise ValueError("sample_size 必须为正，candidate_pool_size 不能小于 10")
    rng = np.random.default_rng(20260820)
    sample_rows = rng.choice(len(documents), size=min(sample_size, len(documents)), replace=False)
    modes = {"text": [], "image": [], "joint": []}

    for raw_doc_row in sample_rows:
        doc_row = int(raw_doc_row)
        text_query = np.asarray(text_embeddings[doc_row : doc_row + 1], dtype="float32")
        first_image_row = int(doc_image_rows[doc_row][0])
        image_query = np.asarray(image_embeddings[first_image_row : first_image_row + 1], dtype="float32")
        for mode in modes:
            active_text = text_query if mode in {"text", "joint"} else None
            active_image = image_query if mode in {"image", "joint"} else None
            exact_scores = exact_all_document_scores(
                text_query=active_text,
                image_query=active_image,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                relation_image_rows=rel_images,
                relation_doc_rows=rel_docs,
            )
            exact = top_rows(exact_scores, max(FINAL_KS))
            candidates = hnsw_candidates(
                text_index, image_index, image_assets, active_text, active_image, candidate_pool_size
            )
            approximate_rows = [
                int(row["doc_row_id"])
                for row in exact_score_documents(
                    candidates,
                    text_query=active_text,
                    image_query=active_image,
                    text_embeddings=text_embeddings,
                    image_embeddings=image_embeddings,
                    documents=documents,
                    image_assets=image_assets,
                    doc_image_rows=doc_image_rows,
                    alpha=0.5,
                )[: max(FINAL_KS)]
            ]
            modes[mode].append({f"recall_at_{k}": recall_at_k(exact, approximate_rows, k) for k in FINAL_KS})

    mode_metrics: dict[str, Any] = {}
    for mode, rows in modes.items():
        mode_metrics[mode] = {
            f"recall_at_{k}": float(np.mean([row[f"recall_at_{k}"] for row in rows]))
            for k in FINAL_KS
        }
    return {
        "schema_version": 2,
        "sample_size": len(sample_rows),
        "candidate_pool_size": candidate_pool_size,
        "ef_search": FIXED_EF_SEARCH,
        "modes": mode_metrics,
        "overall": {
            f"recall_at_{k}": float(np.mean([mode_metrics[mode][f"recall_at_{k}"] for mode in mode_metrics]))
            for k in FINAL_KS
        },
    }


def main() -> int:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    index_dir = Path(args.index_dir or args.embedding_dir)
    output = Path(args.output or embedding_dir / "hnsw_document_benchmark.json")
    report = benchmark_document_recall(
        embedding_dir, index_dir, args.sample_size, args.candidate_pool_size
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
