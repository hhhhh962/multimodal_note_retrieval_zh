"""把已抽取的文本/图片向量构建为持久化 HNSW 内积索引。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import faiss
import numpy as np
from tqdm import tqdm

try:
    from hnsw_final_benchmark import (
        DEFAULT_CANDIDATE_POOL_SIZE,
        DEFAULT_SAMPLE_SIZE,
        FIXED_EF_SEARCH,
        benchmark_final_recall,
    )
except ImportError:
    from src.hnsw_final_benchmark import (
        DEFAULT_CANDIDATE_POOL_SIZE,
        DEFAULT_SAMPLE_SIZE,
        FIXED_EF_SEARCH,
        benchmark_final_recall,
    )

try:
    from model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
    )
    from utils import ensure_dir, load_json
except ImportError:
    from src.model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
    )
    from src.utils import ensure_dir, load_json


DEFAULT_HNSW_M = 32
DEFAULT_EF_CONSTRUCTION = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为文本向量和图片向量构建 HNSW 内积索引")
    parser.add_argument("--embedding_dir", default="outputs/demo_full")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--hnsw_m", type=int, default=DEFAULT_HNSW_M)
    parser.add_argument("--ef_construction", type=int, default=DEFAULT_EF_CONSTRUCTION)
    parser.add_argument("--replace_existing", action="store_true")
    parser.add_argument("--benchmark_sample_size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--candidate_pool_size", type=int, default=DEFAULT_CANDIDATE_POOL_SIZE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_size <= 0:
        raise ValueError("chunk_size 必须是正整数")
    if args.hnsw_m <= 0:
        raise ValueError("hnsw_m 必须是正整数")
    if args.ef_construction <= 0:
        raise ValueError("ef_construction 必须是正整数")
    if args.benchmark_sample_size <= 0:
        raise ValueError("benchmark_sample_size 必须是正整数")
    if args.candidate_pool_size < 10:
        raise ValueError("candidate_pool_size 不能小于 10")


def load_extraction_metadata(embedding_dir: Path) -> tuple[dict, dict]:
    """读取并交叉校验抽取状态与样本元信息。"""

    state_path = embedding_dir / "extract_state.json"
    item_meta_path = embedding_dir / "item_meta.json"
    state = load_json(state_path)
    item_meta = load_json(item_meta_path)
    if not isinstance(state, dict):
        raise FileNotFoundError(f"缺少可用抽取状态：{state_path}")
    if not isinstance(item_meta, dict):
        raise FileNotFoundError(f"缺少可用样本元信息：{item_meta_path}")
    if state.get("complete") is not True:
        raise RuntimeError("向量抽取尚未 complete，禁止构建半成品索引")

    for label, payload in (("extract_state", state), ("item_meta", item_meta)):
        missing = [field for field in REQUIRED_COMPATIBILITY_FIELDS if field not in payload]
        if missing:
            raise MetadataMismatchError(f"{label} 缺少兼容字段：{', '.join(missing)}")
    assert_metadata_compatible(item_meta, state, context="抽取状态与样本元信息")

    count = int(state.get("count", -1))
    dimension = int(state["dimension"])
    items = item_meta.get("items")
    if count < 0 or dimension <= 0:
        raise ValueError("extract_state 的 count/dimension 无效")
    if not isinstance(items, list) or len(items) != count or int(item_meta.get("count", -1)) != count:
        raise MetadataMismatchError("item_meta 的 count/items 数量与 extract_state 不一致")
    if str(state["normalization"]).lower() != "l2":
        raise MetadataMismatchError("HNSW 内积索引只接受 normalization='l2' 的抽取产物")
    if "max_length" in state and "max_length" in item_meta:
        if int(state["max_length"]) != int(item_meta["max_length"]):
            raise MetadataMismatchError("extract_state 与 item_meta 的 max_length 不一致")
    return state, item_meta


def add_embeddings(
    index: faiss.Index,
    embeddings: np.ndarray,
    chunk_size: int,
    desc: str,
    normalization: str = "l2",
) -> None:
    """检查并分块把 float32 向量加入索引。"""

    total = embeddings.shape[0]
    for start in tqdm(range(0, total, chunk_size), desc=desc, dynamic_ncols=True):
        end = min(start + chunk_size, total)
        chunk = np.ascontiguousarray(embeddings[start:end], dtype="float32")
        if not np.isfinite(chunk).all():
            raise ValueError(f"{desc} 包含 NaN 或 Inf，禁止构建索引")
        if normalization == "l2" and len(chunk):
            norms = np.linalg.norm(chunk, axis=1)
            valid = np.isclose(norms, 1.0, rtol=1e-3, atol=1e-4)
            if not np.all(valid):
                first = int(np.flatnonzero(~valid)[0])
                raise ValueError(f"{desc} 第 {start + first} 行不是 l2 归一化向量")
        index.add(chunk)


def is_hnsw_index(index: Any) -> bool:
    """兼容 SWIG 反序列化对象，使用 hnsw 成员识别 HNSW。"""

    return hasattr(index, "hnsw")


def validate_hnsw_index(
    index: Any,
    expected_count: int,
    expected_dimension: int,
    expected_m: int,
    expected_ef_construction: int,
) -> None:
    """校验落盘再加载后的 HNSW 结构和关键参数。"""

    if not is_hnsw_index(index):
        raise MetadataMismatchError(f"索引类型不是 HNSW：{type(index).__name__}")
    if int(index.ntotal) != int(expected_count) or int(index.d) != int(expected_dimension):
        raise MetadataMismatchError(
            f"HNSW count/dim={index.ntotal}/{index.d}，期望 {expected_count}/{expected_dimension}"
        )
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise MetadataMismatchError("HNSW 必须使用 METRIC_INNER_PRODUCT")
    if int(index.hnsw.efConstruction) != int(expected_ef_construction):
        raise MetadataMismatchError("HNSW efConstruction 与构建参数不一致")
    # FAISS 第 0 层默认允许 2*M 个邻居；该值在序列化后仍可读取。
    if int(index.hnsw.nb_neighbors(0)) != int(expected_m) * 2:
        raise MetadataMismatchError("HNSW M 与构建参数不一致")


def build_one(
    embedding_path: Path,
    index_path: Path,
    chunk_size: int,
    hnsw_m: int,
    ef_construction: int,
    expected_count: int | None = None,
    expected_dimension: int | None = None,
    normalization: str = "l2",
) -> dict:
    """为单个向量矩阵构建、保存并重新校验 HNSW。"""

    embeddings = np.load(embedding_path, mmap_mode="r")
    if embeddings.ndim != 2:
        raise ValueError(f"{embedding_path} 必须是二维向量矩阵")
    if expected_count is not None and int(embeddings.shape[0]) != int(expected_count):
        raise MetadataMismatchError(
            f"{embedding_path.name} count={embeddings.shape[0]}，期望 {expected_count}"
        )
    if expected_dimension is not None and int(embeddings.shape[1]) != int(expected_dimension):
        raise MetadataMismatchError(
            f"{embedding_path.name} dimension={embeddings.shape[1]}，期望 {expected_dimension}"
        )

    dim = int(embeddings.shape[1])
    started = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    add_embeddings(index, embeddings, chunk_size, embedding_path.stem, normalization=normalization)
    faiss.write_index(index, str(index_path))
    del index

    persisted = faiss.read_index(str(index_path))
    validate_hnsw_index(persisted, embeddings.shape[0], dim, hnsw_m, ef_construction)
    del persisted
    return {
        "path": str(index_path),
        "count": int(embeddings.shape[0]),
        "dim": dim,
        "dimension": dim,
        "normalization": normalization,
        "size_bytes": int(index_path.stat().st_size),
        "build_seconds": time.perf_counter() - started,
    }


def persist_fixed_ef_search(index_path: Path) -> None:
    """把固定 efSearch=512 写入已构建的 HNSW 文件。"""

    index = faiss.read_index(str(index_path))
    if not is_hnsw_index(index):
        raise MetadataMismatchError(f"索引类型不是 HNSW：{type(index).__name__}")
    index.hnsw.efSearch = FIXED_EF_SEARCH
    faiss.write_index(index, str(index_path))


def write_json_file(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def commit_replacement(staged: Mapping[Path, Path], rollback_paths: Mapping[Path, Path]) -> None:
    """覆盖正式产物；失败时使用短暂回滚文件恢复，成功后立即删除旧文件。"""

    moved_old: list[Path] = []
    moved_new: list[Path] = []
    try:
        for target, rollback in rollback_paths.items():
            if rollback.exists():
                raise FileExistsError(f"发现遗留回滚文件，拒绝继续：{rollback}")
            if target.exists():
                os.replace(target, rollback)
                moved_old.append(target)
        for staged_path, target in staged.items():
            os.replace(staged_path, target)
            moved_new.append(target)
    except Exception:
        for target in reversed(moved_new):
            if target.exists():
                target.unlink()
        for target in reversed(moved_old):
            rollback = rollback_paths[target]
            if rollback.exists():
                os.replace(rollback, target)
        raise
    else:
        for rollback in rollback_paths.values():
            rollback.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    validate_args(args)
    embedding_dir = Path(args.embedding_dir)
    output_dir = ensure_dir(args.output_dir or embedding_dir)
    state, item_meta = load_extraction_metadata(embedding_dir)
    count = int(state["count"])
    dimension = int(state["dimension"])
    normalization = str(state["normalization"]).lower()

    text_target = output_dir / "text.index"
    image_target = output_dir / "image.index"
    meta_target = output_dir / "index_meta.json"
    benchmark_target = output_dir / "hnsw_benchmark.json"
    targets = (text_target, image_target, meta_target)
    existing = [path for path in targets if path.exists()]
    if existing and not args.replace_existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"索引输出已存在；如确认替换请传 --replace_existing：{names}")
    if args.replace_existing and existing and len(existing) != len(targets):
        raise FileNotFoundError("replace_existing 要求旧 text.index、image.index、index_meta.json 全部存在")

    text_temporary = output_dir / ".text.index.hnsw.tmp"
    image_temporary = output_dir / ".image.index.hnsw.tmp"
    meta_temporary = output_dir / ".index_meta.hnsw.tmp"
    benchmark_temporary = output_dir / ".hnsw_benchmark.json.tmp"
    temporaries = (text_temporary, image_temporary, meta_temporary, benchmark_temporary)
    leftovers = [path for path in temporaries if path.exists()]
    if leftovers:
        raise FileExistsError("发现上次遗留的 HNSW 临时产物，请确认无任务运行后删除")

    try:
        text_info = build_one(
            embedding_dir / "text_embeddings.npy",
            text_temporary,
            args.chunk_size,
            args.hnsw_m,
            args.ef_construction,
            expected_count=count,
            expected_dimension=dimension,
            normalization=normalization,
        )
        image_info = build_one(
            embedding_dir / "image_embeddings.npy",
            image_temporary,
            args.chunk_size,
            args.hnsw_m,
            args.ef_construction,
            expected_count=count,
            expected_dimension=dimension,
            normalization=normalization,
        )
        persist_fixed_ef_search(text_temporary)
        persist_fixed_ef_search(image_temporary)
        benchmark = benchmark_final_recall(
            text_temporary,
            image_temporary,
            embedding_dir / "text_embeddings.npy",
            embedding_dir / "image_embeddings.npy",
            item_meta["items"],
            sample_size=args.benchmark_sample_size,
            candidate_pool_size=args.candidate_pool_size,
        )
        text_info["path"] = str(text_target)
        image_info["path"] = str(image_target)
        benchmark["build"] = {
            "text_seconds": text_info["build_seconds"],
            "image_seconds": image_info["build_seconds"],
            "text_size_bytes": int(text_temporary.stat().st_size),
            "image_size_bytes": int(image_temporary.stat().st_size),
        }
        contract = {field: state[field] for field in REQUIRED_COMPATIBILITY_FIELDS}
        index_meta = {
            **contract,
            "count": count,
            "dimension": dimension,
            "normalization": normalization,
            "max_length": state.get("max_length", item_meta.get("max_length")),
            "text": text_info,
            "image": image_info,
            "index_type": "IndexHNSWFlat",
            "metric": "inner_product",
            "hnsw": {
                "m": args.hnsw_m,
                "ef_construction": args.ef_construction,
                "ef_search": FIXED_EF_SEARCH,
            },
            "complete": True,
        }
        write_json_file(index_meta, meta_temporary)
        write_json_file(benchmark, benchmark_temporary)

        staged = {
            text_temporary: text_target,
            image_temporary: image_target,
            meta_temporary: meta_target,
            benchmark_temporary: benchmark_target,
        }
        rollback_paths = {
            text_target: output_dir / ".text.index.rollback",
            image_target: output_dir / ".image.index.rollback",
            meta_target: output_dir / ".index_meta.rollback",
        }
        commit_replacement(staged, rollback_paths)
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)

    print(f"已用 HNSW 替换：{text_target}")
    print(f"已用 HNSW 替换：{image_target}")
    print(f"验收报告：{benchmark_target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
