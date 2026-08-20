"""Document-level candidate expansion, exact scoring, and late fusion."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROUTE_KEYS = ("score_tt", "score_ti", "score_it", "score_ii")


def collect_faiss_scores(distances: np.ndarray, indices: np.ndarray) -> dict[int, float]:
    scores: dict[int, float] = {}
    for score, row_id in zip(distances[0].tolist(), indices[0].tolist()):
        if int(row_id) < 0:
            continue
        row = int(row_id)
        scores[row] = max(float(score), scores.get(row, float("-inf")))
    return scores


def build_doc_image_rows(
    document_count: int, image_assets: Sequence[Mapping[str, Any]]
) -> list[list[int]]:
    rows: list[list[int]] = [[] for _ in range(document_count)]
    for image_row, asset in enumerate(image_assets):
        for doc_row in asset["doc_row_ids"]:
            doc_row = int(doc_row)
            if doc_row < 0 or doc_row >= document_count:
                raise ValueError(f"image row {image_row} 的 doc_row_id={doc_row} 越界")
            rows[doc_row].append(image_row)
    if any(not image_rows for image_rows in rows):
        missing = [row for row, image_rows in enumerate(rows) if not image_rows][:20]
        raise ValueError(f"存在没有图片的文档：{missing}")
    return rows


def expand_image_hits_to_documents(
    image_scores: Mapping[int, float], image_assets: Sequence[Mapping[str, Any]]
) -> tuple[dict[int, float], dict[int, int]]:
    """Map image-row scores to owner documents, keeping the best image per doc."""

    doc_scores: dict[int, float] = {}
    matched_rows: dict[int, int] = {}
    for image_row, score in image_scores.items():
        if image_row < 0 or image_row >= len(image_assets):
            raise ValueError(f"图片索引返回越界 image_row={image_row}")
        for raw_doc_row in image_assets[image_row]["doc_row_ids"]:
            doc_row = int(raw_doc_row)
            if score > doc_scores.get(doc_row, float("-inf")):
                doc_scores[doc_row] = float(score)
                matched_rows[doc_row] = image_row
    return doc_scores, matched_rows


def adaptive_image_candidates(
    index: Any,
    query_embedding: np.ndarray,
    image_assets: Sequence[Mapping[str, Any]],
    target_documents: int,
    *,
    multiplier: int = 4,
) -> tuple[set[int], dict[int, int]]:
    """Expand image HNSW depth until enough unique owner documents are found."""

    if target_documents <= 0 or int(getattr(index, "ntotal", 0)) <= 0:
        return set(), {}
    total = int(index.ntotal)
    depth = min(total, max(target_documents, target_documents * multiplier))
    while True:
        distances, indices = index.search(np.asarray(query_embedding, dtype="float32"), depth)
        image_scores = collect_faiss_scores(distances, indices)
        doc_scores, matched = expand_image_hits_to_documents(image_scores, image_assets)
        if len(doc_scores) >= target_documents or depth >= total:
            return set(doc_scores), matched
        depth = min(total, depth * 2)


def exact_score_documents(
    candidate_doc_rows: Iterable[int],
    *,
    text_query: np.ndarray | None,
    image_query: np.ndarray | None,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    documents: Sequence[Mapping[str, Any]],
    image_assets: Sequence[Mapping[str, Any]],
    doc_image_rows: Sequence[Sequence[int]],
    alpha: float,
) -> list[dict[str, Any]]:
    """Exactly rescore candidate documents over their complete image sets."""

    has_text = text_query is not None
    has_image = image_query is not None
    if not has_text and not has_image:
        raise ValueError("至少需要一种查询向量")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha 必须位于 [0, 1]")
    text_vector = None if text_query is None else np.asarray(text_query, dtype="float32").reshape(-1)
    image_vector = None if image_query is None else np.asarray(image_query, dtype="float32").reshape(-1)
    denominator = int(has_text) + int(has_image)
    results: list[dict[str, Any]] = []

    for doc_row in sorted(set(int(row) for row in candidate_doc_rows)):
        if doc_row < 0 or doc_row >= len(documents):
            raise ValueError(f"候选 doc_row={doc_row} 越界")
        image_rows = list(doc_image_rows[doc_row])
        if not image_rows:
            raise ValueError(f"document {doc_row} 没有图片向量")
        doc_text = np.asarray(text_embeddings[doc_row], dtype="float32")
        doc_images = np.asarray(image_embeddings[image_rows], dtype="float32")
        score_tt = float(doc_text @ text_vector) if has_text else 0.0
        score_it = float(doc_text @ image_vector) if has_image else 0.0
        score_ti_values = doc_images @ text_vector if has_text else None
        score_ii_values = doc_images @ image_vector if has_image else None
        score_ti = float(np.max(score_ti_values)) if score_ti_values is not None else 0.0
        score_ii = float(np.max(score_ii_values)) if score_ii_values is not None else 0.0
        score_text_index = ((score_tt if has_text else 0.0) + (score_it if has_image else 0.0)) / denominator
        score_image_index = ((score_ti if has_text else 0.0) + (score_ii if has_image else 0.0)) / denominator
        score_mm = float(alpha) * score_text_index + (1.0 - float(alpha)) * score_image_index

        matched_values = np.zeros(len(image_rows), dtype="float32")
        if score_ti_values is not None:
            matched_values += score_ti_values
        if score_ii_values is not None:
            matched_values += score_ii_values
        matched_values /= denominator
        best_local = int(np.argmax(matched_values))
        matched_image_row = image_rows[best_local]
        matched_image_path = str(image_assets[matched_image_row]["image_path"])

        document = dict(documents[doc_row])
        image_paths = [str(image["image_path"]) for image in document["images"]]
        document.update(
            {
                "score_tt": score_tt,
                "score_ti": score_ti,
                "score_it": score_it,
                "score_ii": score_ii,
                "score_text_index": score_text_index,
                "score_image_index": score_image_index,
                "score_mm": score_mm,
                "image_paths": image_paths,
                "image_count": len(image_paths),
                "matched_image_path": matched_image_path,
                "best_image_path": matched_image_path,
            }
        )
        results.append(document)
    results.sort(key=lambda row: (-float(row["score_mm"]), int(row["doc_row_id"])))
    return results
