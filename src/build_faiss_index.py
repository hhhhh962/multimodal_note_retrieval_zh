"""Build separate document-text and unique-image HNSW indexes for schema v2."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import faiss
import numpy as np
from tqdm import tqdm

try:
    from document_schema import SCHEMA_VERSION, validate_retrieval_meta
    from model_fingerprint import REQUIRED_COMPATIBILITY_FIELDS, assert_metadata_compatible
except ImportError:
    from src.document_schema import SCHEMA_VERSION, validate_retrieval_meta
    from src.model_fingerprint import REQUIRED_COMPATIBILITY_FIELDS, assert_metadata_compatible


DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200
FIXED_EF_SEARCH = 512
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_CANDIDATE_POOL_SIZE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_dir", default="outputs/finetuned_docs_v2")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--hnsw_m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--ef_construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument(
        "--faiss_threads",
        type=int,
        default=int(os.environ.get("FAISS_NUM_THREADS", "1")),
        help="FAISS/OpenMP 构建线程数；小内存机器默认 1",
    )
    parser.add_argument(
        "--low_memory_build",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="用 FP16 SQ 仅构建 HNSW 图，再装配标准 float32 IndexHNSWFlat",
    )
    parser.add_argument("--replace_existing", action="store_true")
    # Retained for CLI compatibility; benchmarking is now an explicit separate step.
    parser.add_argument("--benchmark_sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate_pool_size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("chunk_size", "hnsw_m", "ef_construction", "faiss_threads"):
        if not hasattr(args, name):
            continue
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} 必须是正整数")
    if hasattr(args, "benchmark_sample_size") and int(args.benchmark_sample_size) <= 0:
        raise ValueError("benchmark_sample_size 必须是正整数")
    if hasattr(args, "candidate_pool_size") and int(args.candidate_pool_size) <= 0:
        raise ValueError("candidate_pool_size 必须是正整数")


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_extraction_metadata(embedding_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_json(embedding_dir / "extract_state.json")
    meta = load_json(embedding_dir / "retrieval_meta.json")
    if not isinstance(state, dict) or not isinstance(meta, dict):
        if (embedding_dir / "item_meta.json").is_file():
            raise ValueError(
                "检测到旧 pair-level 产物；请先运行 scripts/migrate_pair_artifacts.py"
            )
        raise FileNotFoundError("缺少 extract_state.json 或 retrieval_meta.json")
    validate_retrieval_meta(meta, embedding_dir)
    if int(state.get("schema_version", 0)) != SCHEMA_VERSION or not state.get("complete"):
        raise ValueError("extract_state 不是完整的 schema v2 状态")
    assert_metadata_compatible(meta, state, context="抽取状态与 retrieval_meta")
    if meta.get("retrieval_manifest_hash") != state.get("retrieval_manifest_hash"):
        raise ValueError("抽取状态与 retrieval_meta 的清单哈希不一致")
    return state, meta


def add_embeddings(index: faiss.Index, embeddings: np.ndarray, chunk_size: int, desc: str) -> None:
    for start in tqdm(range(0, len(embeddings), chunk_size), desc=desc, dynamic_ncols=True):
        end = min(start + chunk_size, len(embeddings))
        chunk = np.ascontiguousarray(embeddings[start:end], dtype="float32")
        if not np.isfinite(chunk).all():
            raise ValueError(f"{desc} embedding 包含 NaN/Inf，分块起点={start}")
        norms = np.linalg.norm(chunk, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError(f"{desc} 不是 l2 归一化向量，分块起点={start}")
        index.add(chunk)


def build_one(
    embedding_path: Path,
    index_path: Path,
    chunk_size: int,
    hnsw_m: int,
    ef_construction: int,
    expected_count: int | None = None,
    expected_dimension: int | None = None,
    normalization: str = "l2",
    low_memory_build: bool = False,
) -> dict[str, Any]:
    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"{embedding_path.name} 必须是二维矩阵")
    count, dimension = map(int, embeddings.shape)
    if expected_count is not None and count != int(expected_count):
        raise ValueError(f"{embedding_path.name} count={count}，期望 {expected_count}")
    if expected_dimension is not None and dimension != int(expected_dimension):
        raise ValueError(f"{embedding_path.name} dimension={dimension}，期望 {expected_dimension}")
    if str(normalization).lower() != "l2":
        raise ValueError("HNSW 内积索引只支持 l2 归一化向量")
    if low_memory_build:
        index = faiss.IndexHNSWSQ(
            dimension,
            faiss.ScalarQuantizer.QT_fp16,
            int(hnsw_m),
            faiss.METRIC_INNER_PRODUCT,
        )
    else:
        index = faiss.IndexHNSWFlat(dimension, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = FIXED_EF_SEARCH
    started = time.perf_counter()
    add_embeddings(index, embeddings, int(chunk_size), embedding_path.stem)
    if int(index.ntotal) != count:
        raise RuntimeError("HNSW ntotal 与 embedding 数量不一致")
    if low_memory_build:
        index = assemble_flat_hnsw(
            index,
            embeddings,
            dimension,
            hnsw_m,
            count,
            index_path.with_name(f".{embedding_path.stem}.hnsw_graph.tmp"),
        )
    faiss.write_index(index, str(index_path))
    del index
    gc.collect()
    loaded = faiss.read_index(str(index_path))
    if int(loaded.ntotal) != count or int(loaded.d) != dimension:
        raise RuntimeError("HNSW 重载后的 count/dimension 不一致")
    loaded.hnsw.efSearch = FIXED_EF_SEARCH
    faiss.write_index(loaded, str(index_path))
    del loaded
    gc.collect()
    return {
        "count": count,
        "dimension": dimension,
        "normalization": "l2",
        "size_bytes": int(index_path.stat().st_size),
        "build_seconds": time.perf_counter() - started,
        "graph_build_storage": "sq_fp16" if low_memory_build else "flat_float32",
        "final_storage": "flat_float32",
    }


def assemble_flat_hnsw(
    graph_index: faiss.Index,
    embeddings: np.ndarray,
    dimension: int,
    hnsw_m: int,
    count: int,
    scratch_dir: Path,
) -> faiss.Index:
    """Move an SQ-built graph into a serializable float32 IndexHNSWFlat."""

    vector_fields = (
        "levels",
        "offsets",
        "neighbors",
        "cum_nneighbor_per_level",
        "assign_probas",
    )
    scalar_fields = (
        "entry_point",
        "max_level",
        "efConstruction",
        "efSearch",
        "check_relative_distance",
        "search_bounded_queue",
    )
    if scratch_dir.exists():
        raise FileExistsError(f"发现遗留 HNSW 图临时目录：{scratch_dir}")
    scratch_dir.mkdir(parents=True)
    try:
        graph_scalars = {name: getattr(graph_index.hnsw, name) for name in scalar_fields}
        graph_storage = faiss.downcast_index(graph_index.storage)
        graph_storage.reset()
        del graph_storage
        trim_allocator()
        print("临时 SQ 向量已释放，正在把 HNSW 邻接图逐字段转存到磁盘", flush=True)
        for name in vector_fields:
            values = faiss.vector_to_array(getattr(graph_index.hnsw, name))
            np.save(scratch_dir / f"{name}.npy", values)
            del values
            trim_allocator()
        del graph_index
        trim_allocator()

        print("正在装配标准 float32 IndexHNSWFlat", flush=True)
        final = faiss.IndexHNSWFlat(dimension, int(hnsw_m), faiss.METRIC_INNER_PRODUCT)
        storage = faiss.downcast_index(final.storage)
        for start in range(0, count, 5000):
            end = min(start + 5000, count)
            storage.add(np.ascontiguousarray(embeddings[start:end], dtype="float32"))
        if int(storage.ntotal) != count:
            raise RuntimeError("装配后的 IndexFlat 存储数量不一致")
        final.ntotal = count
        for name in vector_fields:
            values = np.load(scratch_dir / f"{name}.npy", mmap_mode="r")
            faiss.copy_array_to_vector(values, getattr(final.hnsw, name))
            del values
            trim_allocator()
        for name, value in graph_scalars.items():
            setattr(final.hnsw, name, value)
        if int(final.hnsw.offsets.size()) != count + 1:
            raise RuntimeError("装配后的 HNSW offsets 数量不一致")
        return final
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def trim_allocator() -> None:
    """Return freed C/C++ heap pages to the OS when glibc exposes malloc_trim."""

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def main() -> int:
    args = parse_args()
    validate_args(args)
    faiss.omp_set_num_threads(int(args.faiss_threads))
    embedding_dir = Path(args.embedding_dir)
    output_dir = Path(args.output_dir or args.embedding_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state, meta = load_extraction_metadata(embedding_dir)
    dimension = int(meta["dimension"])
    document_count = int(meta["document_count"])
    image_count = int(meta["image_count"])

    targets = {
        "text": output_dir / "text.index",
        "image": output_dir / "image.index",
        "meta": output_dir / "index_meta.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.replace_existing:
        raise FileExistsError("索引产物已存在；确认替换时传 --replace_existing")
    if args.replace_existing and existing and len(existing) != len(targets):
        raise FileNotFoundError("replace_existing 要求旧 text.index、image.index、index_meta.json 全部存在")

    temporary = {
        "text": output_dir / ".text.index.v2.tmp",
        "image": output_dir / ".image.index.v2.tmp",
        "meta": output_dir / ".index_meta.v2.tmp",
    }
    if any(path.exists() for path in temporary.values()):
        raise FileExistsError("发现遗留的 v2 索引临时文件")

    try:
        text_info = build_one(
            embedding_dir / "text_embeddings.npy",
            temporary["text"],
            args.chunk_size,
            args.hnsw_m,
            args.ef_construction,
            expected_count=document_count,
            expected_dimension=dimension,
            normalization=meta["normalization"],
            low_memory_build=args.low_memory_build,
        )
        image_info = build_one(
            embedding_dir / "image_embeddings.npy",
            temporary["image"],
            args.chunk_size,
            args.hnsw_m,
            args.ef_construction,
            expected_count=image_count,
            expected_dimension=dimension,
            normalization=meta["normalization"],
            low_memory_build=args.low_memory_build,
        )
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    contract = {field: meta[field] for field in REQUIRED_COMPATIBILITY_FIELDS}
    index_meta = {
        **contract,
        "schema_version": SCHEMA_VERSION,
        "retrieval_manifest_hash": meta["retrieval_manifest_hash"],
        "document_count": document_count,
        "image_count": image_count,
        "dimension": dimension,
        "normalization": "l2",
        "max_length": meta.get("max_length", state.get("max_length")),
        "text": text_info,
        "image": image_info,
        "index_type": "IndexHNSWFlat",
        "metric": "inner_product",
        "hnsw": {
            "m": int(args.hnsw_m),
            "ef_construction": int(args.ef_construction),
            "ef_search": FIXED_EF_SEARCH,
        },
        "complete": True,
    }
    write_json(temporary["meta"], index_meta)

    backups: dict[Path, Path] = {}
    try:
        for name, target in targets.items():
            if target.exists():
                backup = target.with_suffix(target.suffix + ".v2.rollback")
                if backup.exists():
                    raise FileExistsError(f"遗留回滚文件：{backup}")
                os.replace(target, backup)
                backups[target] = backup
            os.replace(temporary[name], target)
    except Exception:
        for name, target in targets.items():
            if target.exists() and target not in backups:
                target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    print(json.dumps(index_meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
