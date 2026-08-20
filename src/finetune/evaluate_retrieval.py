"""Chinese-CLIP LoRA 的检索评估入口。

本模块同时提供两层接口：底层 NumPy 函数用于精确、可复现地计算多正样本
Recall@1/5/10；高层 ``evaluate_checkpoint`` 直接供训练循环调用，并能从三套
官方数据的 JSONL 标注与图片 LMDB 编码验证或测试候选池。

评估时文本和图片都按唯一 ID 编码。一个 query 的任意正确候选进入前 K 即视为命中，
因此一图多文和一文多图不会被错误地按单一对角线答案计分。MUGE test 标注中的
``image_ids=[]`` 会被识别为无答案，只写出排序文件，不产生伪 Recall。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np


DEFAULT_KS = (1, 5, 10)
FIVE_TASKS = (
    "muge_t2i",
    "flickr_t2i",
    "flickr_i2t",
    "coco_t2i",
    "coco_i2t",
)


def _as_uid(value: Any) -> str:
    """把数值或字符串 ID 统一成稳定的字符串键。"""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _canonical_dataset_name(name: str) -> str:
    """把配置中的数据集别名折叠成任务名使用的三个标准名称。"""

    compact = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if "muge" in compact:
        return "muge"
    if "flickr" in compact:
        return "flickr"
    if "coco" in compact:
        return "coco"
    return compact or "dataset"


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    """将二维特征矩阵转为连续 float32 并执行安全的 L2 归一化。"""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"特征必须是二维矩阵，实际形状为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("特征矩阵包含 NaN 或 Inf")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("特征矩阵包含零向量，无法进行余弦检索")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _validate_unique_ids(ids: Sequence[str], label: str) -> tuple[str, ...]:
    """标准化 ID，并阻止同一候选在评估矩阵中重复出现。"""

    result = tuple(_as_uid(value) for value in ids)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} 中存在重复 ID，评估前必须按 ID 去重")
    return result


@dataclass
class RetrievalEmbeddings:
    """一套数据集的唯一文本、唯一图片、完整关系与对应特征。"""

    name: str
    split: str
    text_embeddings: np.ndarray
    image_embeddings: np.ndarray
    text_uids: Sequence[str]
    image_uids: Sequence[str]
    text_to_images: Mapping[str, Iterable[str]]
    image_to_texts: Mapping[str, Iterable[str]] | None = None
    text_payloads: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    include_image_to_text: bool = True

    def __post_init__(self) -> None:
        """校验矩阵行号、ID 和关系，并补齐反向多正样本映射。"""

        self.text_embeddings = _normalise_rows(self.text_embeddings)
        self.image_embeddings = _normalise_rows(self.image_embeddings)
        self.text_uids = _validate_unique_ids(self.text_uids, "text_uids")
        self.image_uids = _validate_unique_ids(self.image_uids, "image_uids")
        if len(self.text_uids) != self.text_embeddings.shape[0]:
            raise ValueError("text_uids 数量与文本特征行数不一致")
        if len(self.image_uids) != self.image_embeddings.shape[0]:
            raise ValueError("image_uids 数量与图片特征行数不一致")
        if self.text_embeddings.shape[1] != self.image_embeddings.shape[1]:
            raise ValueError("文本与图片特征维度不一致")
        self.text_to_images = {
            _as_uid(text_uid): frozenset(_as_uid(image_uid) for image_uid in image_uids)
            for text_uid, image_uids in self.text_to_images.items()
        }
        if self.image_to_texts is None:
            reverse: MutableMapping[str, set[str]] = {}
            for text_uid, image_uids in self.text_to_images.items():
                for image_uid in image_uids:
                    reverse.setdefault(image_uid, set()).add(text_uid)
            self.image_to_texts = {key: frozenset(values) for key, values in reverse.items()}
        else:
            self.image_to_texts = {
                _as_uid(image_uid): frozenset(_as_uid(text_uid) for text_uid in text_uids)
                for image_uid, text_uids in self.image_to_texts.items()
            }

    @property
    def dimension(self) -> int:
        """返回共享投影空间维度。"""

        return int(self.text_embeddings.shape[1])


def _topk_indices(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    query_batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """分块执行精确内积检索，并以候选行号作为同分时的稳定次序。"""

    queries = _normalise_rows(query_embeddings)
    candidates = _normalise_rows(candidate_embeddings)
    if queries.shape[1] != candidates.shape[1]:
        raise ValueError("query 与 candidate 特征维度不一致")
    if candidates.shape[0] == 0:
        raise ValueError("候选池为空")
    keep = min(max(int(top_k), 1), candidates.shape[0])
    all_indices = np.empty((queries.shape[0], keep), dtype=np.int64)
    all_scores = np.empty((queries.shape[0], keep), dtype=np.float32)
    candidate_rows = np.arange(candidates.shape[0], dtype=np.int64)
    for start in range(0, queries.shape[0], max(int(query_batch_size), 1)):
        end = min(start + max(int(query_batch_size), 1), queries.shape[0])
        scores = np.asarray(queries[start:end] @ candidates.T, dtype=np.float32)
        for local_row, row_scores in enumerate(scores):
            if keep == candidates.shape[0]:
                selected = candidate_rows
            else:
                threshold = np.partition(row_scores, -keep)[-keep]
                above = np.flatnonzero(row_scores > threshold)
                tied = np.flatnonzero(row_scores == threshold)
                selected = np.concatenate((above, tied[: keep - len(above)]))
            order = np.lexsort((selected, -row_scores[selected]))
            ranked = selected[order]
            all_indices[start + local_row] = ranked
            all_scores[start + local_row] = row_scores[ranked]
    return all_indices, all_scores


def recall_at_k(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    query_uids: Sequence[str],
    candidate_uids: Sequence[str],
    positives: Mapping[str, Iterable[str]],
    ks: Sequence[int] = DEFAULT_KS,
    query_batch_size: int = 256,
    return_rankings: bool = False,
) -> dict[str, Any]:
    """计算多正样本 Recall@K；没有有效答案的 query 不进入分母。"""

    query_ids = _validate_unique_ids(query_uids, "query_uids")
    candidate_ids = _validate_unique_ids(candidate_uids, "candidate_uids")
    if len(query_ids) != np.asarray(query_embeddings).shape[0]:
        raise ValueError("query_uids 数量与 query 特征行数不一致")
    if len(candidate_ids) != np.asarray(candidate_embeddings).shape[0]:
        raise ValueError("candidate_uids 数量与 candidate 特征行数不一致")
    clean_ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not clean_ks:
        raise ValueError("ks 至少需要一个正整数")
    candidate_set = set(candidate_ids)
    labelled_rows: list[int] = []
    labelled_positives: list[frozenset[str]] = []
    missing_answer_queries: list[str] = []
    for row, query_uid in enumerate(query_ids):
        answers = frozenset(_as_uid(value) for value in positives.get(query_uid, ()))
        available_answers = answers & candidate_set
        if available_answers:
            labelled_rows.append(row)
            labelled_positives.append(available_answers)
        else:
            missing_answer_queries.append(query_uid)
    result: dict[str, Any] = {
        "num_queries": len(labelled_rows),
        "num_queries_without_answers": len(missing_answer_queries),
        "candidate_count": len(candidate_ids),
        "ks": list(clean_ks),
        "scored": bool(labelled_rows),
    }
    if not labelled_rows:
        for k in clean_ks:
            result[f"recall_at_{k}"] = None
        result["mean_recall"] = None
        if return_rankings:
            indices, scores = _topk_indices(
                query_embeddings,
                candidate_embeddings,
                max(clean_ks),
                query_batch_size=query_batch_size,
            )
            result["ranking_indices"] = indices
            result["ranking_scores"] = scores
        return result
    labelled_embeddings = np.asarray(query_embeddings)[np.asarray(labelled_rows, dtype=np.int64)]
    indices, scores = _topk_indices(
        labelled_embeddings,
        candidate_embeddings,
        max(clean_ks),
        query_batch_size=query_batch_size,
    )
    recalls: list[float] = []
    for k in clean_ks:
        hits = 0
        effective_k = min(k, indices.shape[1])
        for query_row, answers in enumerate(labelled_positives):
            retrieved = {candidate_ids[index] for index in indices[query_row, :effective_k]}
            hits += int(bool(retrieved & answers))
        value = hits / len(labelled_rows)
        result[f"recall_at_{k}"] = float(value)
        recalls.append(float(value))
    result["mean_recall"] = float(np.mean(recalls))
    if return_rankings:
        result["ranking_query_uids"] = [query_ids[row] for row in labelled_rows]
        result["ranking_indices"] = indices
        result["ranking_scores"] = scores
    return result


def _direction_metrics(
    dataset: RetrievalEmbeddings,
    direction: str,
    candidate_embeddings: np.ndarray | None = None,
    candidate_uids: Sequence[str] | None = None,
    positives: Mapping[str, Iterable[str]] | None = None,
    query_batch_size: int = 256,
) -> dict[str, Any]:
    """按指定方向选择 query/candidate，并调用统一 Recall 实现。"""

    if direction == "text_to_image":
        query_embeddings = dataset.text_embeddings
        query_uids = dataset.text_uids
        default_candidates = dataset.image_embeddings
        default_candidate_uids = dataset.image_uids
        default_positives = dataset.text_to_images
    elif direction == "image_to_text":
        query_embeddings = dataset.image_embeddings
        query_uids = dataset.image_uids
        default_candidates = dataset.text_embeddings
        default_candidate_uids = dataset.text_uids
        default_positives = dataset.image_to_texts or {}
    else:
        raise ValueError(f"未知检索方向：{direction}")
    metrics = recall_at_k(
        query_embeddings=query_embeddings,
        candidate_embeddings=default_candidates if candidate_embeddings is None else candidate_embeddings,
        query_uids=query_uids,
        candidate_uids=default_candidate_uids if candidate_uids is None else candidate_uids,
        positives=default_positives if positives is None else positives,
        query_batch_size=query_batch_size,
    )
    metrics["direction"] = direction
    return metrics


def evaluate_dataset(dataset: RetrievalEmbeddings, query_batch_size: int = 256) -> dict[str, Any]:
    """计算单数据集指标；MUGE 默认只计文搜图。"""

    directions = ["text_to_image"]
    if dataset.include_image_to_text:
        directions.append("image_to_text")
    result: dict[str, Any] = {
        "name": dataset.name,
        "canonical_name": _canonical_dataset_name(dataset.name),
        "split": dataset.split,
        "dimension": dataset.dimension,
        "text_count": len(dataset.text_uids),
        "image_count": len(dataset.image_uids),
    }
    scored_means: list[float] = []
    for direction in directions:
        metrics = _direction_metrics(dataset, direction, query_batch_size=query_batch_size)
        result[direction] = metrics
        if metrics["mean_recall"] is not None:
            scored_means.append(float(metrics["mean_recall"]))
    result["dataset_mean_recall"] = float(np.mean(scored_means)) if scored_means else None
    return result


def _namespaced(dataset_name: str, uid: str) -> str:
    """为合并候选池增加来源前缀，彻底隔离跨数据集同名 ID。"""

    return f"{_canonical_dataset_name(dataset_name)}::{_as_uid(uid)}"


def evaluate_merged_candidate_pool(
    datasets: Sequence[RetrievalEmbeddings],
    query_batch_size: int = 256,
) -> dict[str, Any]:
    """让每套数据的 query 面对三套数据拼接后的候选，测量跨域干扰。"""

    if not datasets:
        return {"datasets": {}, "task_macro_mean": None}
    dimensions = {dataset.dimension for dataset in datasets}
    if len(dimensions) != 1:
        raise ValueError("合并候选池要求所有数据集的特征维度一致")
    all_images = np.concatenate([dataset.image_embeddings for dataset in datasets], axis=0)
    all_texts = np.concatenate([dataset.text_embeddings for dataset in datasets], axis=0)
    all_image_uids = tuple(
        _namespaced(dataset.name, uid) for dataset in datasets for uid in dataset.image_uids
    )
    all_text_uids = tuple(
        _namespaced(dataset.name, uid) for dataset in datasets for uid in dataset.text_uids
    )
    report: dict[str, Any] = {
        "image_candidate_count": len(all_image_uids),
        "text_candidate_count": len(all_text_uids),
        "datasets": {},
    }
    task_means: list[float] = []
    for dataset in datasets:
        prefix_text_ids = tuple(_namespaced(dataset.name, uid) for uid in dataset.text_uids)
        prefix_image_ids = tuple(_namespaced(dataset.name, uid) for uid in dataset.image_uids)
        text_relations = {
            _namespaced(dataset.name, text_uid): {
                _namespaced(dataset.name, image_uid) for image_uid in image_uids
            }
            for text_uid, image_uids in dataset.text_to_images.items()
        }
        image_relations = {
            _namespaced(dataset.name, image_uid): {
                _namespaced(dataset.name, text_uid) for text_uid in text_uids
            }
            for image_uid, text_uids in (dataset.image_to_texts or {}).items()
        }
        t2i = recall_at_k(
            dataset.text_embeddings,
            all_images,
            prefix_text_ids,
            all_image_uids,
            text_relations,
            query_batch_size=query_batch_size,
        )
        entry: dict[str, Any] = {"text_to_image": t2i}
        if t2i["mean_recall"] is not None:
            task_means.append(float(t2i["mean_recall"]))
        if dataset.include_image_to_text:
            i2t = recall_at_k(
                dataset.image_embeddings,
                all_texts,
                prefix_image_ids,
                all_text_uids,
                image_relations,
                query_batch_size=query_batch_size,
            )
            entry["image_to_text"] = i2t
            if i2t["mean_recall"] is not None:
                task_means.append(float(i2t["mean_recall"]))
        report["datasets"][_canonical_dataset_name(dataset.name)] = entry
    report["task_macro_mean"] = float(np.mean(task_means)) if task_means else None
    return report


def _task_name(dataset_name: str, direction: str) -> str:
    """把数据集方向转换为固定的模型选择任务键。"""

    suffix = "t2i" if direction == "text_to_image" else "i2t"
    return f"{_canonical_dataset_name(dataset_name)}_{suffix}"


def evaluate_suite(
    datasets: Sequence[RetrievalEmbeddings],
    include_merged_pool: bool = True,
    query_batch_size: int = 256,
) -> dict[str, Any]:
    """汇总三数据集、五任务宏平均与可选的跨域合并候选池结果。"""

    dataset_reports: dict[str, Any] = {}
    task_reports: dict[str, Any] = {}
    for dataset in datasets:
        canonical = _canonical_dataset_name(dataset.name)
        report = evaluate_dataset(dataset, query_batch_size=query_batch_size)
        dataset_reports[canonical] = report
        for direction in ("text_to_image", "image_to_text"):
            metrics = report.get(direction)
            if metrics is None:
                continue
            task_reports[_task_name(dataset.name, direction)] = metrics
    five_values = [
        float(task_reports[name]["mean_recall"])
        for name in FIVE_TASKS
        if name in task_reports and task_reports[name].get("mean_recall") is not None
    ]
    all_five_scored = len(five_values) == len(FIVE_TASKS)
    result: dict[str, Any] = {
        "datasets": dataset_reports,
        "tasks": task_reports,
        "five_task_names": list(FIVE_TASKS),
        "five_task_macro_mean": float(np.mean(five_values)) if all_five_scored else None,
        "selection_score": float(np.mean(five_values)) if all_five_scored else None,
        "eligible_for_selection": all_five_scored,
    }
    if include_merged_pool:
        result["merged_candidate_pool"] = evaluate_merged_candidate_pool(
            datasets,
            query_batch_size=query_batch_size,
        )
    return result


def assess_acceptance(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate_demo: Mapping[str, float] | None = None,
    baseline_demo: Mapping[str, float] | None = None,
    max_dataset_drop: float = 0.02,
    max_ndcg_drop: float = 0.01,
    max_latency_increase: float = 0.10,
) -> dict[str, Any]:
    """按部署门槛比较微调 checkpoint 与同口径基座报告。"""

    candidate_macro = candidate.get("five_task_macro_mean")
    baseline_macro = baseline.get("five_task_macro_mean")
    macro_improved = (
        candidate_macro is not None
        and baseline_macro is not None
        and float(candidate_macro) > float(baseline_macro)
    )
    improved_t2i_r1: list[str] = []
    t2i_comparisons: dict[str, Any] = {}
    dataset_drop_checks: dict[str, Any] = {}
    for dataset_name in ("muge", "flickr", "coco"):
        candidate_dataset = candidate.get("datasets", {}).get(dataset_name, {})
        baseline_dataset = baseline.get("datasets", {}).get(dataset_name, {})
        candidate_r1 = candidate_dataset.get("text_to_image", {}).get("recall_at_1")
        baseline_r1 = baseline_dataset.get("text_to_image", {}).get("recall_at_1")
        improved = (
            candidate_r1 is not None
            and baseline_r1 is not None
            and float(candidate_r1) > float(baseline_r1)
        )
        if improved:
            improved_t2i_r1.append(dataset_name)
        t2i_comparisons[dataset_name] = {
            "candidate": candidate_r1,
            "baseline": baseline_r1,
            "improved": improved,
        }
        candidate_mean = candidate_dataset.get("dataset_mean_recall")
        baseline_mean = baseline_dataset.get("dataset_mean_recall")
        drop = None
        passed = False
        if candidate_mean is not None and baseline_mean is not None:
            drop = float(baseline_mean) - float(candidate_mean)
            passed = drop <= float(max_dataset_drop) + 1e-12
        dataset_drop_checks[dataset_name] = {
            "candidate": candidate_mean,
            "baseline": baseline_mean,
            "drop": drop,
            "passed": passed,
        }
    demo_checks: dict[str, Any] = {"provided": False, "passed": None}
    if candidate_demo is not None and baseline_demo is not None:
        candidate_ndcg = candidate_demo.get("ndcg_at_10")
        baseline_ndcg = baseline_demo.get("ndcg_at_10")
        candidate_latency = candidate_demo.get("avg_latency_ms")
        baseline_latency = baseline_demo.get("avg_latency_ms")
        ndcg_passed = (
            candidate_ndcg is not None
            and baseline_ndcg is not None
            and float(candidate_ndcg) >= float(baseline_ndcg) - float(max_ndcg_drop)
        )
        latency_limit = None
        latency_passed = False
        if baseline_latency is not None:
            latency_limit = float(baseline_latency) * (1.0 + float(max_latency_increase))
            latency_passed = candidate_latency is not None and float(candidate_latency) <= latency_limit
        demo_checks = {
            "provided": True,
            "ndcg_passed": ndcg_passed,
            "latency_passed": latency_passed,
            "latency_limit_ms": latency_limit,
            "passed": bool(ndcg_passed and latency_passed),
        }
    core_passed = bool(
        macro_improved
        and len(improved_t2i_r1) >= 2
        and all(check["passed"] for check in dataset_drop_checks.values())
    )
    deployment_passed = core_passed and (demo_checks["passed"] is True)
    return {
        "macro_improved": macro_improved,
        "candidate_five_task_macro_mean": candidate_macro,
        "baseline_five_task_macro_mean": baseline_macro,
        "t2i_r1": t2i_comparisons,
        "t2i_r1_improved_datasets": improved_t2i_r1,
        "at_least_two_t2i_r1_improved": len(improved_t2i_r1) >= 2,
        "dataset_mean_drop": dataset_drop_checks,
        "core_retrieval_passed": core_passed,
        "demo_regression": demo_checks,
        "deployment_passed": deployment_passed,
        "deployment_decision_complete": demo_checks["provided"],
    }


def write_muge_submission(
    output_path: str | Path,
    text_uids: Sequence[str],
    image_uids: Sequence[str],
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    top_k: int = 10,
    query_batch_size: int = 256,
) -> Path:
    """为无答案的 MUGE test 生成官方 schema 的文搜图 JSONL。"""

    text_ids = _validate_unique_ids(text_uids, "text_uids")
    image_ids = _validate_unique_ids(image_uids, "image_uids")
    indices, _ = _topk_indices(
        text_embeddings,
        image_embeddings,
        top_k=top_k,
        query_batch_size=query_batch_size,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text_uid, rows in zip(text_ids, indices):
            payload = {
                "text_id": text_uid.split(":", 1)[-1],
                "image_ids": [image_ids[int(row)].split(":", 1)[-1] for row in rows],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def _load_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """读取 YAML/JSON 配置；训练调用时也可直接传 Mapping。"""

    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        result = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("读取 YAML 配置需要安装 pyyaml") from exc
        result = yaml.safe_load(text)
    if not isinstance(result, Mapping):
        raise ValueError(f"配置根节点必须是对象：{path}")
    result = dict(result)
    result.setdefault("_config_dir", str(path.resolve().parent))
    return result


def _dataset_configs(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """兼容 ``datasets`` 与 ``data.datasets`` 两种自然配置层级。"""

    direct = config.get("datasets")
    if isinstance(direct, Mapping):
        return direct
    data = config.get("data")
    if isinstance(data, Mapping):
        nested = data.get("datasets")
        if isinstance(nested, Mapping):
            return nested
        likely = {
            key: value
            for key, value in data.items()
            if isinstance(value, Mapping) and _canonical_dataset_name(key) in {"muge", "flickr", "coco"}
        }
        if likely:
            return likely
    raise ValueError("配置中缺少 datasets 或 data.datasets")


def _resolve_path(value: str | Path, config: Mapping[str, Any]) -> Path:
    """相对路径优先相对配置文件目录解析。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    base = Path(str(config.get("_config_dir", ".")))
    return (base / path).resolve()


def _dataset_root(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    split: str | None = None,
) -> Path:
    """从数据集配置读取原始数据根目录。"""

    value = spec.get("root") or spec.get("data_dir") or spec.get("path")
    if value is not None:
        return _resolve_path(value, config)
    split_names = [split] if split else ["valid", "test", "train"]
    for split_name in split_names:
        configured = spec.get(f"{split_name}_lmdb")
        if configured is None:
            continue
        lmdb_path = _resolve_path(configured, config)
        if lmdb_path.name == "imgs":
            lmdb_path = lmdb_path.parent
        if lmdb_path.parent.name == "lmdb":
            return lmdb_path.parent.parent
    raise ValueError("每个评估数据集必须配置 root/data_dir/path，或提供可反推根目录的 {split}_lmdb")


def _split_value(value: Any, split: str) -> Any:
    """若字段是按 split 映射，取出当前 split；否则原样返回。"""

    if isinstance(value, Mapping):
        return value.get(split)
    return value


def _annotation_path(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    split: str,
) -> Path:
    """定位 ``{split}_texts.jsonl`` 完整关系标注。"""

    configured = (
        _split_value(spec.get("annotations"), split)
        or _split_value(spec.get("texts"), split)
        or spec.get(f"{split}_texts")
    )
    if configured is not None:
        return _resolve_path(configured, config)
    return _dataset_root(spec, config, split=split) / f"{split}_texts.jsonl"


def _image_lmdb_path(
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    split: str,
) -> Path:
    """定位当前 split 的只读图片 LMDB。"""

    configured = (
        _split_value(spec.get("image_lmdb"), split)
        or _split_value(spec.get("lmdb"), split)
        or spec.get(f"{split}_lmdb")
    )
    if configured is not None:
        path = _resolve_path(configured, config)
        return path / "imgs" if (path / "imgs").is_dir() else path
    return _dataset_root(spec, config, split=split) / "lmdb" / split / "imgs"


def _read_annotations(path: Path, source: str) -> tuple[list[str], list[str], dict[str, frozenset[str]], dict[str, dict[str, Any]]]:
    """读取唯一文本和完整一文多图关系，保留原始 test 文本 ID。"""

    text_uids: list[str] = []
    texts: list[str] = []
    relations: dict[str, frozenset[str]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "text_id" not in obj or "text" not in obj or "image_ids" not in obj:
                raise ValueError(f"{path}:{line_number} 缺少 text_id/text/image_ids")
            uid = f"{source}:{_as_uid(obj['text_id'])}"
            if uid in relations:
                raise ValueError(f"{path}:{line_number} 重复 text_id={obj['text_id']}")
            image_uids = frozenset(f"{source}:{_as_uid(value)}" for value in obj["image_ids"])
            text_uids.append(uid)
            texts.append(str(obj["text"]))
            relations[uid] = image_uids
            payloads[uid] = dict(obj)
    return text_uids, texts, relations, payloads


def _iter_lmdb_images(path: Path, source: str) -> Iterator[tuple[str, Any]]:
    """顺序读取图片 LMDB，跳过 ``num_images`` 元数据键并解码为 RGB。"""

    try:
        import lmdb
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("从官方图片 LMDB 评估需要安装 lmdb 和 pillow") from exc
    environment = lmdb.open(
        str(path),
        readonly=True,
        create=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=64,
    )
    try:
        with environment.begin(buffers=False) as transaction:
            with transaction.cursor() as cursor:
                for raw_key, raw_value in cursor:
                    if raw_key == b"num_images":
                        continue
                    image_id = raw_key.decode("utf-8", errors="strict")
                    encoded = raw_value.decode("utf-8", errors="ignore")
                    encoded += "=" * (-len(encoded) % 4)
                    image = Image.open(BytesIO(base64.urlsafe_b64decode(encoded))).convert("RGB")
                    yield f"{source}:{image_id}", image
    finally:
        environment.close()


def _batched(values: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    """把任意可迭代对象按固定大小分批，不缓存完整图片集合。"""

    batch: list[Any] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _model_device(model: Any, requested: str | Any | None = None) -> Any:
    """选择评估设备；显式值优先，否则沿用模型当前设备。"""

    import torch

    if requested is not None:
        return torch.device(requested)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _feature_tensor(value: Any, preferred: str) -> Any:
    """兼容 transformers 不同版本的特征返回对象。"""

    import torch

    if isinstance(value, torch.Tensor):
        return value
    for attr in (preferred, "text_embeds", "image_embeds", "pooler_output"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    raise TypeError(f"无法从 {type(value)!r} 提取特征张量")


def _feature_method(model: Any, method_name: str) -> Any:
    """在 PEFT 包装和底层模型中寻找双塔特征接口。"""

    candidates = [model]
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            candidates.append(getter())
        except Exception:
            pass
    candidates.append(getattr(getattr(model, "base_model", None), "model", None))
    for candidate in candidates:
        method = getattr(candidate, method_name, None) if candidate is not None else None
        if callable(method):
            return method
    raise AttributeError(f"PEFT/ChineseCLIP 模型均未暴露 {method_name}")


def _autocast_context(device: Any, precision: str = "auto") -> Any:
    """CUDA 上按 BF16 优先策略启用 autocast，CPU 始终保持 FP32。"""

    import torch

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type != "cuda":
        return nullcontext()
    requested = str(precision).lower()
    if requested in {"fp32", "float32", "none"}:
        return nullcontext()
    if requested in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("evaluation.precision=bf16，但当前 CUDA 设备不支持 BF16")
        dtype = torch.bfloat16
    elif requested in {"fp16", "float16"}:
        dtype = torch.float16
    elif requested == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        raise ValueError(f"未知 evaluation.precision={precision!r}")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _encode_texts(
    model: Any,
    processor: Any,
    texts: Sequence[str],
    device: Any,
    batch_size: int,
    max_length: int,
    precision: str = "auto",
) -> np.ndarray:
    """使用 Hugging Face ChineseCLIP 文本塔编码唯一文本。"""

    import torch

    outputs: list[np.ndarray] = []
    for batch in _batched(texts, max(int(batch_size), 1)):
        encoded = processor(
            text=batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), _autocast_context(device, precision):
            value = _feature_method(model, "get_text_features")(**encoded)
        tensor = _feature_tensor(value, "text_embeds")
        outputs.append(tensor.detach().float().cpu().numpy())
    if not outputs:
        dimension = int(getattr(getattr(model, "config", None), "projection_dim", 512))
        return np.empty((0, dimension), dtype=np.float32)
    return _normalise_rows(np.concatenate(outputs, axis=0))


def _encode_images(
    model: Any,
    processor: Any,
    images: Iterable[tuple[str, Any]],
    device: Any,
    batch_size: int,
    precision: str = "auto",
) -> tuple[list[str], np.ndarray]:
    """顺序编码唯一 LMDB 图片，并及时关闭 PIL 对象降低内存占用。"""

    import torch

    image_uids: list[str] = []
    outputs: list[np.ndarray] = []
    for batch in _batched(images, max(int(batch_size), 1)):
        batch_ids = [item[0] for item in batch]
        batch_images = [item[1] for item in batch]
        try:
            encoded = processor(images=batch_images, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode(), _autocast_context(device, precision):
                value = _feature_method(model, "get_image_features")(**encoded)
            tensor = _feature_tensor(value, "image_embeds")
            outputs.append(tensor.detach().float().cpu().numpy())
            image_uids.extend(batch_ids)
        finally:
            for image in batch_images:
                close = getattr(image, "close", None)
                if callable(close):
                    close()
    if not outputs:
        dimension = int(getattr(getattr(model, "config", None), "projection_dim", 512))
        return image_uids, np.empty((0, dimension), dtype=np.float32)
    return image_uids, _normalise_rows(np.concatenate(outputs, axis=0))


def encode_configured_dataset(
    model: Any,
    processor: Any,
    config: Mapping[str, Any],
    name: str,
    spec: Mapping[str, Any],
    split: str,
    device: Any = None,
) -> RetrievalEmbeddings:
    """从原始 JSONL 与只读图片 LMDB 编码一套完整评估候选池。"""

    source = str(spec.get("source") or name)
    annotation_path = _annotation_path(spec, config, split)
    image_lmdb_path = _image_lmdb_path(spec, config, split)
    if not annotation_path.is_file():
        raise FileNotFoundError(f"缺少评估标注：{annotation_path}")
    if not image_lmdb_path.is_dir():
        raise FileNotFoundError(f"缺少图片 LMDB：{image_lmdb_path}")
    text_uids, texts, text_to_images, payloads = _read_annotations(annotation_path, source)
    evaluation = config.get("evaluation", {}) if isinstance(config.get("evaluation"), Mapping) else {}
    run_device = _model_device(model, device)
    precision = str(evaluation.get("precision", "auto"))
    text_embeddings = _encode_texts(
        model,
        processor,
        texts,
        run_device,
        int(evaluation.get("text_batch_size", 256)),
        int(evaluation.get("max_length", 52)),
        precision,
    )
    image_uids, image_embeddings = _encode_images(
        model,
        processor,
        _iter_lmdb_images(image_lmdb_path, source),
        run_device,
        int(evaluation.get("image_batch_size", 64)),
        precision,
    )
    candidate_images = set(image_uids)
    empty_queries = [uid for uid, answers in text_to_images.items() if not answers]
    missing_references = {
        uid: sorted(set(answers) - candidate_images)
        for uid, answers in text_to_images.items()
        if set(answers) - candidate_images
    }
    canonical_name = _canonical_dataset_name(name)
    is_unlabelled_muge_test = (
        canonical_name == "muge"
        and split == "test"
        and len(empty_queries) == len(text_uids)
        and not missing_references
    )
    if not is_unlabelled_muge_test and empty_queries:
        preview = ", ".join(empty_queries[:5])
        raise ValueError(
            f"{name} {split} 有 {len(empty_queries)} 条 query 缺少 GT；仅允许 MUGE test 全部 image_ids=[]。示例：{preview}"
        )
    if missing_references:
        preview = "; ".join(
            f"{uid}->{values[:3]}" for uid, values in list(missing_references.items())[:5]
        )
        raise ValueError(
            f"{name} {split} 有 {len(missing_references)} 条 query 引用了候选池不存在的图片：{preview}"
        )
    include_i2t = bool(spec.get("image_to_text", _canonical_dataset_name(name) != "muge"))
    return RetrievalEmbeddings(
        name=name,
        split=split,
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        text_uids=text_uids,
        image_uids=image_uids,
        text_to_images=text_to_images,
        text_payloads=payloads,
        include_image_to_text=include_i2t,
    )


def evaluate_checkpoint(
    model: Any,
    processor: Any,
    config: str | Path | Mapping[str, Any],
    split: str = "valid",
    device: Any = None,
) -> dict[str, Any]:
    """训练循环调用入口：编码三套 split，并返回固定五任务模型选择报告。"""

    loaded_config = _load_config(config)
    specifications = _dataset_configs(loaded_config)
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    datasets: list[RetrievalEmbeddings] = []
    try:
        for name, spec in specifications.items():
            if not isinstance(spec, Mapping) or spec.get("enabled", True) is False:
                continue
            datasets.append(
                encode_configured_dataset(
                    model=model,
                    processor=processor,
                    config=loaded_config,
                    name=str(name),
                    spec=spec,
                    split=split,
                    device=device,
                )
            )
        evaluation = loaded_config.get("evaluation", {})
        include_merged = not isinstance(evaluation, Mapping) or bool(evaluation.get("merged_candidate_pool", True))
        query_batch_size = int(evaluation.get("similarity_query_batch_size", 256)) if isinstance(evaluation, Mapping) else 256
        report = evaluate_suite(
            datasets,
            include_merged_pool=include_merged,
            query_batch_size=query_batch_size,
        )
        report["split"] = split
        if split == "test":
            output_dir = Path(str(evaluation.get("output_dir", "outputs/finetune/evaluation"))) if isinstance(evaluation, Mapping) else Path("outputs/finetune/evaluation")
            for dataset in datasets:
                if _canonical_dataset_name(dataset.name) != "muge":
                    continue
                if any(dataset.text_to_images.get(uid) for uid in dataset.text_uids):
                    continue
                submission_path = write_muge_submission(
                    output_dir / "muge_test_predictions.jsonl",
                    dataset.text_uids,
                    dataset.image_uids,
                    dataset.text_embeddings,
                    dataset.image_embeddings,
                    top_k=int(evaluation.get("submission_top_k", 10)) if isinstance(evaluation, Mapping) else 10,
                    query_batch_size=query_batch_size,
                )
                report.setdefault("prediction_exports", {})["muge"] = str(submission_path)
        return report
    finally:
        if was_training and hasattr(model, "train"):
            model.train()


def _json_ready(value: Any) -> Any:
    """把 NumPy 标量和路径递归转换成 JSON 可写类型。"""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """读取独立零样本/微调 checkpoint 评估参数。"""

    parser = argparse.ArgumentParser(description="评估 Chinese-CLIP 三数据集检索 Recall")
    parser.add_argument("--config", default="configs/finetune_lora.yaml")
    parser.add_argument("--model", default=None, help="完整 HF 模型目录或模型名")
    parser.add_argument("--adapter", default=None, help="可选 PEFT adapter/checkpoint 目录")
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def _load_model_for_cli(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[Any, Any]:
    """按需加载 HF 基座和 PEFT adapter，避免纯指标单测触发外部依赖。"""

    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    model_config = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    model_name = args.model or model_config.get("name") or model_config.get("base_model") or "OFA-Sys/chinese-clip-vit-base-patch16"
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    model = ChineseCLIPModel.from_pretrained(model_name)
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("加载 LoRA adapter 需要安装 peft") from exc
        adapter_path = Path(args.adapter)
        checkpoint_dir = adapter_path if (adapter_path / "adapter").is_dir() else adapter_path.parent
        adapter_dir = checkpoint_dir / "adapter" if (checkpoint_dir / "adapter").is_dir() else adapter_path
        model = PeftModel.from_pretrained(model, adapter_dir)
        trainable_state = checkpoint_dir / "trainable_state.pt"
        if trainable_state.is_file():
            try:
                state = torch.load(trainable_state, map_location="cpu", weights_only=False)
            except TypeError:
                state = torch.load(trainable_state, map_location="cpu")
            try:
                from src.finetune.checkpointing import restore_extra_trainable_state
            except ImportError:
                from finetune.checkpointing import restore_extra_trainable_state
            restore_extra_trainable_state(model, state)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device).eval()
    return model, processor


def main(argv: Sequence[str] | None = None) -> int:
    """执行独立评估并把完整报告写为 JSON。"""

    args = parse_args(argv)
    config = _load_config(args.config)
    model, processor = _load_model_for_cli(args, config)
    report = evaluate_checkpoint(model, processor, config, split=args.split)
    evaluation = config.get("evaluation", {}) if isinstance(config.get("evaluation"), Mapping) else {}
    output = Path(args.output or evaluation.get("report_path", f"outputs/finetune/{args.split}_metrics.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_json_ready(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
