"""评估固定 efSearch=512 的最终融合、MUGE 聚合后 Recall@5/10。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import faiss
import numpy as np

try:
    from muge_aggregation import aggregate_results, load_muge_mapping
except ImportError:
    from src.muge_aggregation import aggregate_results, load_muge_mapping


BENCHMARK_SEED = 42
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_CANDIDATE_POOL_SIZE = 100
DEFAULT_ALPHA = 0.5
FIXED_EF_SEARCH = 512
FINAL_KS = (5, 10)
MODE_WEIGHTS = {
    "text": {"tt": 0.5, "ti": 0.5},
    "image": {"it": 0.5, "ii": 0.5},
    "joint": {"tt": 0.25, "ti": 0.25, "it": 0.25, "ii": 0.25},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估现有 HNSW 的最终融合 Recall@5/10")
    parser.add_argument("--embedding_dir", default="outputs/finetuned_full")
    parser.add_argument("--index_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate_pool_size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    return parser.parse_args()


def final_ranked_keys(
    route_scores: Mapping[str, Mapping[int, float]],
    items: Sequence[Mapping[str, Any]],
    row_to_text_id: Mapping[str, str],
    mode: str,
    top_k: int,
) -> list[str]:
    """按线上公式融合四路候选、排序、MUGE 一文多图聚合，返回 doc_id 列表。"""

    if mode not in MODE_WEIGHTS:
        raise ValueError(f"未知查询模式：{mode}")
    weights = MODE_WEIGHTS[mode]
    candidates = sorted(set().union(*(route_scores[route].keys() for route in weights)))
    fused: list[dict[str, Any]] = []
    for row_id in candidates:
        if row_id < 0 or row_id >= len(items):
            raise ValueError(f"索引返回越界 row_id={row_id}")
        score_mm = sum(
            weight * float(route_scores[route].get(row_id, 0.0))
            for route, weight in weights.items()
        )
        fused.append({
            "row_id": row_id,
            "score_mm": score_mm,
            "score_tt": float(route_scores.get("tt", {}).get(row_id, 0.0)),
            "score_ti": float(route_scores.get("ti", {}).get(row_id, 0.0)),
            "score_it": float(route_scores.get("it", {}).get(row_id, 0.0)),
            "score_ii": float(route_scores.get("ii", {}).get(row_id, 0.0)),
            "score_text_index": score_mm,
            "score_image_index": score_mm,
        })
    fused.sort(key=lambda row: row["score_mm"], reverse=True)
    # 取前 top_k*2 个图文对作为聚合候选池，保证聚合后有足够文档
    candidates_pool = fused[: max(top_k * 2, top_k)]
    aggregated = aggregate_results(candidates_pool, items, row_to_text_id)
    return [row["doc_id"] for row in aggregated[:top_k]]


def recall_at_k(exact: Sequence[str], approximate: Sequence[str], k: int) -> float:
    """计算一条查询最终 Top-K 的集合 Recall（文档级，doc_id 为单位）。"""

    if len(exact) < k or len(approximate) < k:
        raise ValueError(f"最终聚合结果不足 Top-{k}：exact={len(exact)}, approximate={len(approximate)}")
    return len(set(exact[:k]) & set(approximate[:k])) / float(k)


def ids_scores_to_mapping(ids: np.ndarray, scores: np.ndarray) -> dict[int, float]:
    """把一次 FAISS 返回值转换为线上流水线使用的 row_id 到分数字典。"""

    return {int(row_id): float(score) for score, row_id in zip(scores, ids) if int(row_id) >= 0}


def add_to_flat(index: faiss.Index, embeddings: np.ndarray, chunk_size: int = 50000) -> None:
    """分块加入内存 Flat，避免创建超大连续临时副本。"""

    for start in range(0, len(embeddings), chunk_size):
        end = min(start + chunk_size, len(embeddings))
        chunk = np.ascontiguousarray(embeddings[start:end], dtype="float32")
        if not np.isfinite(chunk).all():
            raise ValueError(f"embedding 包含 NaN/Inf，首个异常分块起点={start}")
        index.add(chunk)


def search_batched(index: faiss.Index, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    scores, ids = index.search(np.ascontiguousarray(queries, dtype="float32"), top_k)
    return scores, ids


def route_row(scores: np.ndarray, ids: np.ndarray, row: int) -> dict[int, float]:
    return ids_scores_to_mapping(ids[row], scores[row])


def benchmark_final_recall(
    text_index_path: Path,
    image_index_path: Path,
    text_embeddings_path: Path,
    image_embeddings_path: Path,
    items: Sequence[Mapping[str, Any]],
    row_to_text_id: Mapping[str, str],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> dict:
    """以临时内存 Flat 为真值，评估固定 efSearch 的三种最终检索结果。"""

    if sample_size <= 0 or candidate_pool_size < max(FINAL_KS):
        raise ValueError("sample_size 必须为正，candidate_pool_size 不能小于 10")

    text_embeddings = np.load(text_embeddings_path, mmap_mode="r")
    image_embeddings = np.load(image_embeddings_path, mmap_mode="r")
    if text_embeddings.ndim != 2 or image_embeddings.ndim != 2:
        raise ValueError("文本和图片 embedding 必须是二维矩阵")
    if text_embeddings.shape != image_embeddings.shape:
        raise ValueError("文本和图片 embedding 的形状必须一致")
    count, dimension = map(int, text_embeddings.shape)
    if len(items) != count:
        raise ValueError(f"item_meta 数量={len(items)}，embedding 数量={count}")

    text_hnsw = faiss.read_index(str(text_index_path))
    image_hnsw = faiss.read_index(str(image_index_path))
    for label, index in (("text", text_hnsw), ("image", image_hnsw)):
        if not hasattr(index, "hnsw"):
            raise ValueError(f"{label} 索引不是 HNSW")
        if int(index.ntotal) != count or int(index.d) != dimension:
            raise ValueError(f"{label} HNSW 数量或维度与 embedding 不一致")
        if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise ValueError(f"{label} HNSW 必须使用内积度量")
        index.hnsw.efSearch = FIXED_EF_SEARCH
    hnsw_m = int(text_hnsw.hnsw.nb_neighbors(0)) // 2
    ef_construction = int(text_hnsw.hnsw.efConstruction)
    if int(image_hnsw.hnsw.nb_neighbors(0)) // 2 != hnsw_m:
        raise ValueError("文本和图片 HNSW 的 M 不一致")
    if int(image_hnsw.hnsw.efConstruction) != ef_construction:
        raise ValueError("文本和图片 HNSW 的 efConstruction 不一致")

    exact_text = faiss.IndexFlatIP(dimension)
    exact_image = faiss.IndexFlatIP(dimension)
    add_to_flat(exact_text, text_embeddings)
    add_to_flat(exact_image, image_embeddings)

    sample_size = min(int(sample_size), count)
    candidate_pool_size = min(int(candidate_pool_size), count)
    rng = np.random.default_rng(BENCHMARK_SEED)
    rows = rng.choice(count, size=sample_size, replace=False)
    text_queries = np.ascontiguousarray(text_embeddings[rows], dtype="float32")
    image_queries = np.ascontiguousarray(image_embeddings[rows], dtype="float32")

    exact_arrays = {
        "tt": search_batched(exact_text, text_queries, candidate_pool_size),
        "ti": search_batched(exact_image, text_queries, candidate_pool_size),
        "it": search_batched(exact_text, image_queries, candidate_pool_size),
        "ii": search_batched(exact_image, image_queries, candidate_pool_size),
    }
    recall_values = {mode: {k: [] for k in FINAL_KS} for mode in MODE_WEIGHTS}
    latency_values = {mode: [] for mode in MODE_WEIGHTS}

    for query_index in range(sample_size):
        exact_routes = {
            route: route_row(scores, ids, query_index)
            for route, (scores, ids) in exact_arrays.items()
        }
        exact_final = {
            mode: final_ranked_keys(exact_routes, items, row_to_text_id, mode, max(FINAL_KS))
            for mode in MODE_WEIGHTS
        }

        started = time.perf_counter()
        tt_scores, tt_ids = text_hnsw.search(text_queries[query_index : query_index + 1], candidate_pool_size)
        ti_scores, ti_ids = image_hnsw.search(text_queries[query_index : query_index + 1], candidate_pool_size)
        text_routes = {
            "tt": ids_scores_to_mapping(tt_ids[0], tt_scores[0]),
            "ti": ids_scores_to_mapping(ti_ids[0], ti_scores[0]),
        }
        text_final = final_ranked_keys(text_routes, items, row_to_text_id, "text", max(FINAL_KS))
        latency_values["text"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        it_scores, it_ids = text_hnsw.search(image_queries[query_index : query_index + 1], candidate_pool_size)
        ii_scores, ii_ids = image_hnsw.search(image_queries[query_index : query_index + 1], candidate_pool_size)
        image_routes = {
            "it": ids_scores_to_mapping(it_ids[0], it_scores[0]),
            "ii": ids_scores_to_mapping(ii_ids[0], ii_scores[0]),
        }
        image_final = final_ranked_keys(image_routes, items, row_to_text_id, "image", max(FINAL_KS))
        latency_values["image"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        joint_routes: dict[str, dict[int, float]] = {}
        for route, index, query in (
            ("tt", text_hnsw, text_queries),
            ("ti", image_hnsw, text_queries),
            ("it", text_hnsw, image_queries),
            ("ii", image_hnsw, image_queries),
        ):
            scores, ids = index.search(query[query_index : query_index + 1], candidate_pool_size)
            joint_routes[route] = ids_scores_to_mapping(ids[0], scores[0])
        joint_final = final_ranked_keys(joint_routes, items, row_to_text_id, "joint", max(FINAL_KS))
        latency_values["joint"].append((time.perf_counter() - started) * 1000.0)

        approximate_final = {"text": text_final, "image": image_final, "joint": joint_final}
        for mode in MODE_WEIGHTS:
            for k in FINAL_KS:
                recall_values[mode][k].append(recall_at_k(exact_final[mode], approximate_final[mode], k))
        if (query_index + 1) % 100 == 0 or query_index + 1 == sample_size:
            print(f"最终融合评估 {query_index + 1}/{sample_size}", flush=True)

    modes: dict[str, dict[str, float]] = {}
    all_latencies: list[float] = []
    for mode in MODE_WEIGHTS:
        latencies = np.asarray(latency_values[mode], dtype="float64")
        all_latencies.extend(latency_values[mode])
        modes[mode] = {
            "recall_at_5": float(np.mean(recall_values[mode][5])),
            "recall_at_10": float(np.mean(recall_values[mode][10])),
            "mean_ms": float(np.mean(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
        }
    all_latency_array = np.asarray(all_latencies, dtype="float64")
    return {
        "seed": BENCHMARK_SEED,
        "sample_size": sample_size,
        "candidate_pool_size": candidate_pool_size,
        "final_k": list(FINAL_KS),
        "alpha": DEFAULT_ALPHA,
        "hnsw": {
            "m": hnsw_m,
            "ef_construction": ef_construction,
            "ef_search": FIXED_EF_SEARCH,
        },
        "modes": modes,
        "overall": {
            "recall_at_5": float(np.mean([modes[mode]["recall_at_5"] for mode in MODE_WEIGHTS])),
            "recall_at_10": float(np.mean([modes[mode]["recall_at_10"] for mode in MODE_WEIGHTS])),
            "mean_ms": float(np.mean(all_latency_array)),
            "p95_ms": float(np.percentile(all_latency_array, 95)),
        },
        "latency_note": "仅统计 CPU HNSW 每路 Top-100、最终融合和去重；不包含查询编码。",
        "complete": True,
    }


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return payload


def write_report_atomic(report: Mapping[str, Any], output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(dict(report), file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    embedding_dir = Path(args.embedding_dir)
    index_dir = Path(args.index_dir or args.embedding_dir)
    output = Path(args.output) if args.output else index_dir / "hnsw_benchmark.json"
    item_meta = load_json(embedding_dir / "item_meta.json")
    index_meta = load_json(index_dir / "index_meta.json")
    muge_mapping = load_muge_mapping(embedding_dir / "muge_mapping.json")
    row_to_text_id = muge_mapping["row_to_text_id"]
    hnsw_meta = index_meta.get("hnsw") or {}
    if int(hnsw_meta.get("ef_search", -1)) != FIXED_EF_SEARCH:
        raise ValueError("index_meta.hnsw.ef_search 必须固定为 512")
    report = benchmark_final_recall(
        index_dir / "text.index",
        index_dir / "image.index",
        embedding_dir / "text_embeddings.npy",
        embedding_dir / "image_embeddings.npy",
        item_meta.get("items") or [],
        row_to_text_id,
        sample_size=args.sample_size,
        candidate_pool_size=args.candidate_pool_size,
    )
    write_report_atomic(report, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"已写入：{output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
