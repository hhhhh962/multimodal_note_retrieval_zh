"""
人工 query 评估脚本。

这个脚本分两步使用：先批量运行 manual_queries.jsonl 里的 query 生成候选结果，再读取带人工标签的结果
计算 Precision@5、Precision@10、NDCG@5、NDCG@10 和检索耗时。
"""

"""引入命令行参数解析工具。"""
import argparse
"""引入 csv 工具，用来输出方便表格查看的文件。"""
import csv
"""引入结构化文本工具，用来读写 json/jsonl。"""
import json
"""引入数学工具，用来计算 NDCG。"""
import math
"""引入系统退出工具，用来返回脚本状态码。"""
import sys
"""引入计时工具，用来统计单 query 检索耗时。"""
import time
"""引入路径对象，方便处理输入输出路径。"""
from pathlib import Path
"""引入类型标注，方便阅读数据结构。"""
from typing import Any, Dict, Iterable, List, Tuple

"""引入四路召回搜索流水线。"""
from search_pipeline import DEFAULT_EMBEDDING_DIR, SearchPipeline


FIELDNAMES = [
    "query_id",
    "query",
    "rank",
    "doc_id",
    "image_count",
    "text",
    "best_image_path",
    "image_paths",
    "source",
    "split",
    "score_tt",
    "score_ti",
    "score_text_index",
    "score_image_index",
    "score_mm",
    "latency_ms",
    "label",
    "label_reason",
]


BAD_CASE_FIELDNAMES = [
    "query_id",
    "query",
    "rank",
    "doc_id",
    "image_count",
    "text",
    "best_image_path",
    "score_mm",
    "score_text_index",
    "score_image_index",
    "label",
    "label_reason",
    "bad_case_type",
    "bad_case_reason",
]


BAD_QUERY_FIELDNAMES = [
    "query_id",
    "query",
    "precision_at_5",
    "precision_at_10",
    "ndcg_at_5",
    "ndcg_at_10",
    "avg_latency_ms",
    "bad_case_count",
    "false_positive_count",
    "weak_top_rank_count",
    "relevant_ranked_low_count",
    "analysis",
]


def parse_args() -> argparse.Namespace:
    """
    读取命令行参数。
    """
    """创建顶层命令行解析器。"""
    parser = argparse.ArgumentParser(description="项目一 manual_queries 人工评估工具")
    """创建子命令解析器。"""
    subparsers = parser.add_subparsers(dest="command", required=True)

    """声明 collect 子命令。"""
    collect = subparsers.add_parser("collect", help="批量运行 query 并生成待标注候选")
    """声明 query 清单路径。"""
    collect.add_argument("--queries", default="data/processed/demo/manual_queries.jsonl")
    """声明候选 jsonl 输出路径。"""
    collect.add_argument("--output", default="reports/manual_eval_candidates.jsonl")
    """声明候选 csv 输出路径；为空时自动从 jsonl 路径推导。"""
    collect.add_argument("--output_csv", default=None)
    """声明索引和元信息目录。"""
    collect.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)
    """声明项目根目录。"""
    collect.add_argument("--project_root", default=".")
    """声明模型名称或本地模型目录。"""
    collect.add_argument("--model_name", default=None)
    """声明最多读取多少个 query，默认读取全部。"""
    collect.add_argument("--limit", type=int, default=None)
    """声明每一路召回数量。"""
    collect.add_argument("--top_k_recall", type=int, default=100)
    """声明每个 query 最终保留数量。"""
    collect.add_argument("--top_k_final", type=int, default=10)
    """声明 late fusion 固定系数。"""
    collect.add_argument("--alpha", type=float, default=0.5)
    """声明运行设备；auto 优先 CUDA。"""
    collect.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    """声明是否在显卡上用半精度编码 query。"""
    collect.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    """声明 score 子命令。"""
    score = subparsers.add_parser("score", help="读取带标签结果并计算指标")
    """声明带标签 jsonl 输入路径。"""
    score.add_argument("--labeled", default="reports/manual_eval_labeled.jsonl")
    """声明整体指标 json 输出路径。"""
    score.add_argument("--output_json", default="reports/manual_eval_metrics.json")
    """声明逐 query 指标 csv 输出路径。"""
    score.add_argument("--output_csv", default="reports/manual_eval_per_query.csv")
    """声明摘要 markdown 输出路径。"""
    score.add_argument("--output_md", default="reports/manual_eval_summary.md")
    """声明 bad cases 明细 jsonl 输出路径。"""
    score.add_argument("--output_bad_cases_jsonl", default="reports/manual_eval_bad_cases.jsonl")
    """声明 bad cases 明细 csv 输出路径。"""
    score.add_argument("--output_bad_cases_csv", default="reports/manual_eval_bad_cases.csv")
    """声明 bad query 汇总 csv 输出路径。"""
    score.add_argument("--output_bad_queries_csv", default="reports/manual_eval_bad_queries.csv")
    """声明 bad cases 分析 markdown 输出路径。"""
    score.add_argument("--output_bad_cases_md", default="reports/manual_eval_bad_cases.md")
    """声明 NDCG 低于多少时把 query 纳入重点 bad query。"""
    score.add_argument("--bad_ndcg_threshold", type=float, default=0.85)
    """解析并返回参数。"""
    return parser.parse_args()


def ensure_parent(path: str | Path) -> Path:
    """
    确保输出文件所在目录存在。
    """
    """把路径转成 Path 对象。"""
    path = Path(path)
    """创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    """返回路径对象。"""
    return path


def read_jsonl(path: str | Path, limit: int | None = None) -> List[dict]:
    """
    读取 jsonl 文件。
    """
    """准备结果列表。"""
    rows = []
    """打开文件逐行解析。"""
    with Path(path).open("r", encoding="utf-8") as f:
        """逐行读取。"""
        for line in f:
            """跳过空行。"""
            if not line.strip():
                """继续读取下一行。"""
                continue
            """解析当前行。"""
            rows.append(json.loads(line))
            """达到读取上限时停止。"""
            if limit is not None and len(rows) >= limit:
                """跳出循环。"""
                break
    """返回全部记录。"""
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """
    写出 jsonl 文件。
    """
    """确保输出目录存在。"""
    path = ensure_parent(path)
    """打开输出文件。"""
    with path.open("w", encoding="utf-8") as f:
        """逐行写出。"""
        for row in rows:
            """写出一条结构化记录。"""
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: List[dict], fieldnames: List[str]) -> None:
    """
    写出 csv 文件。
    """
    """确保输出目录存在。"""
    path = ensure_parent(path)
    """打开输出文件。"""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        """创建字典写入器。"""
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        """写出表头。"""
        writer.writeheader()
        """写出数据。"""
        writer.writerows(rows)


def rounded(value: Any, digits: int = 6) -> Any:
    """
    把浮点数统一保留固定小数，其他值原样返回。
    """
    """浮点数做四舍五入。"""
    if isinstance(value, float):
        """返回保留小数后的值。"""
        return round(value, digits)
    """非浮点数原样返回。"""
    return value


def candidate_row(query_obj: dict, rank: int, result: dict, latency_ms: float) -> dict:
    """
    把文档级搜索结果整理成待标注记录。
    """
    """按固定字段构造输出行。"""
    return {
        "query_id": query_obj.get("query_id"),
        "query": query_obj.get("query"),
        "rank": rank,
        "doc_id": result.get("doc_id"),
        "image_count": result.get("image_count", len(result.get("image_paths", []))),
        "text": result.get("text"),
        "best_image_path": result.get("best_image_path"),
        "image_paths": json.dumps(result.get("image_paths", []), ensure_ascii=False),
        "source": result.get("source"),
        "split": result.get("split"),
        "score_tt": rounded(float(result.get("score_tt", 0.0))),
        "score_ti": rounded(float(result.get("score_ti", 0.0))),
        "score_text_index": rounded(float(result.get("score_text_index", 0.0))),
        "score_image_index": rounded(float(result.get("score_image_index", 0.0))),
        "score_mm": rounded(float(result.get("score_mm", 0.0))),
        "latency_ms": rounded(latency_ms, 3),
        "label": None,
        "label_reason": "",
    }


def command_collect(args: argparse.Namespace) -> int:
    """
    批量执行 query 检索，生成待标注候选文件。
    """
    """读取 query 清单。"""
    queries = read_jsonl(args.queries, limit=args.limit)
    """创建并常驻搜索流水线，初始化时间不计入单 query 检索耗时。"""
    pipeline = SearchPipeline(
        embedding_dir=args.embedding_dir,
        model_name=args.model_name,
        project_root=args.project_root,
        device=args.device,
        fp16=args.fp16,
    )
    """准备候选结果列表。"""
    rows = []
    """逐个 query 执行检索。"""
    for index, query_obj in enumerate(queries, start=1):
        """读取当前 query 文本。"""
        query = str(query_obj.get("query", "")).strip()
        """记录搜索开始时间。"""
        start = time.perf_counter()
        """执行搜索。"""
        results = pipeline.search(
            query=query,
            alpha=args.alpha,
            top_k_recall=args.top_k_recall,
            top_k_final=args.top_k_final,
        )
        """计算当前 query 的检索耗时。"""
        latency_ms = (time.perf_counter() - start) * 1000
        """把 topK 结果展开成多行。"""
        for rank, result in enumerate(results, start=1):
            """追加一条候选记录。"""
            rows.append(candidate_row(query_obj, rank, result, latency_ms))
        """输出进度，便于长任务观察。"""
        print(f"已完成 {index}/{len(queries)}: {query}，耗时 {latency_ms:.1f} ms", flush=True)
    """写出 jsonl 候选文件。"""
    write_jsonl(args.output, rows)
    """确定 csv 输出路径。"""
    output_csv = args.output_csv or str(Path(args.output).with_suffix(".csv"))
    """写出 csv 候选文件。"""
    write_csv(output_csv, rows, FIELDNAMES)
    """打印完成信息。"""
    print(f"已写入：{args.output}")
    """打印完成信息。"""
    print(f"已写入：{output_csv}")
    """正常结束。"""
    return 0


def validate_labels(rows: List[dict]) -> None:
    """
    校验所有结果都已标注。
    """
    """逐条检查 label。"""
    for row in rows:
        """读取 label。"""
        label = row.get("label")
        """缺失 label 时直接报错。"""
        if label is None or label == "":
            """抛出包含 query 和 rank 的错误。"""
            raise ValueError(f"缺少 label: {row.get('query_id')} rank={row.get('rank')}")
        """尝试把 label 转成整数。"""
        label = int(label)
        """label 必须是 0、1、2。"""
        if label not in {0, 1, 2}:
            """抛出清晰错误。"""
            raise ValueError(f"非法 label={label}: {row.get('query_id')} rank={row.get('rank')}")
        """把标准化后的 label 写回。"""
        row["label"] = label


def group_by_query(rows: List[dict]) -> Dict[str, List[dict]]:
    """
    按 query_id 分组。
    """
    """准备分组字典。"""
    grouped: Dict[str, List[dict]] = {}
    """逐条加入分组。"""
    for row in rows:
        """读取 query_id。"""
        query_id = str(row.get("query_id"))
        """加入当前 query 的结果列表。"""
        grouped.setdefault(query_id, []).append(row)
    """每组按 rank 排序。"""
    for group in grouped.values():
        """根据 rank 升序排序。"""
        group.sort(key=lambda item: int(item.get("rank", 0)))
    """返回分组结果。"""
    return grouped


def precision_at(rows: List[dict], k: int) -> float:
    """
    计算单 query 的 Precision@K。
    """
    """截取前 K 条。"""
    top_rows = rows[:k]
    """没有结果时返回零。"""
    if not top_rows:
        """返回零。"""
        return 0.0
    """label 大于等于一视为相关。"""
    hits = sum(1 for row in top_rows if int(row["label"]) >= 1)
    """返回相关比例。"""
    return hits / k


def dcg(labels: List[int]) -> float:
    """
    计算 DCG。
    """
    """准备累计值。"""
    total = 0.0
    """逐个位置计算折损收益。"""
    for index, label in enumerate(labels, start=1):
        """计算当前位置收益。"""
        gain = (2 ** label - 1) / math.log2(index + 1)
        """累加收益。"""
        total += gain
    """返回 DCG。"""
    return total


def ndcg_at(rows: List[dict], k: int) -> float:
    """
    计算单 query 的 NDCG@K。
    """
    """读取前 K 条 label。"""
    labels = [int(row["label"]) for row in rows[:k]]
    """当前排序的 DCG。"""
    actual = dcg(labels)
    """理想排序的 DCG。"""
    ideal = dcg(sorted(labels, reverse=True))
    """如果理想值为零，说明前 K 都不相关。"""
    if ideal == 0:
        """返回零。"""
        return 0.0
    """返回归一化结果。"""
    return actual / ideal


def mean(values: List[float]) -> float:
    """
    计算平均值。
    """
    """空列表返回零。"""
    if not values:
        """返回零。"""
        return 0.0
    """返回算术平均。"""
    return sum(values) / len(values)


def percentile(values: List[float], quantile: float) -> float:
    """使用 nearest-rank 定义计算分位数。"""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0, math.ceil(float(quantile) * len(ordered)) - 1)
    return ordered[position]


def analyze_candidate_bad_case(row: dict) -> Tuple[str, str] | None:
    """
    判断单条候选是否属于 bad case。
    """
    """读取排序位置。"""
    rank = int(row.get("rank", 0))
    """读取人工标签。"""
    label = int(row.get("label", 0))
    """明显不相关结果属于强 bad case。"""
    if label == 0:
        """返回错误类型和原因。"""
        return "false_positive", "结果明显不相关，却进入了最终 top10。"
    """前 5 位只达到弱相关，说明前排排序质量还有提升空间。"""
    if rank <= 5 and label == 1:
        """返回错误类型和原因。"""
        return "weak_top_rank", "结果只部分相关，但排在前 5 位。"
    """高相关结果排在 6 到 10 位，说明排序没有把强相关结果充分提前。"""
    if rank > 5 and label == 2:
        """返回错误类型和原因。"""
        return "relevant_ranked_low", "高相关结果排在第 6 名之后，存在排序提前空间。"
    """其余候选不作为 bad case 输出。"""
    return None


def build_bad_cases(rows: List[dict]) -> List[dict]:
    """
    生成候选级 bad cases 明细。
    """
    """准备输出列表。"""
    bad_cases = []
    """逐条候选判断问题类型。"""
    for row in rows:
        """分析当前候选是否属于 bad case。"""
        analysis = analyze_candidate_bad_case(row)
        """没有问题则跳过。"""
        if analysis is None:
            """继续下一条。"""
            continue
        """拆出问题类型和原因。"""
        bad_case_type, bad_case_reason = analysis
        """追加标准化 bad case 记录。"""
        bad_cases.append(
            {
                "query_id": row.get("query_id"),
                "query": row.get("query"),
                "rank": row.get("rank"),
                "doc_id": row.get("doc_id"),
                "image_count": row.get("image_count"),
                "text": row.get("text"),
                "best_image_path": row.get("best_image_path"),
                "score_mm": row.get("score_mm"),
                "score_text_index": row.get("score_text_index"),
                "score_image_index": row.get("score_image_index"),
                "label": row.get("label"),
                "label_reason": row.get("label_reason"),
                "bad_case_type": bad_case_type,
                "bad_case_reason": bad_case_reason,
            }
        )
    """返回全部 bad case。"""
    return bad_cases


def bad_query_analysis(row: dict, type_counts: Dict[str, int], threshold: float) -> str:
    """
    为单个 bad query 生成一句可读分析。
    """
    """读取核心指标。"""
    p5 = float(row["precision_at_5"])
    p10 = float(row["precision_at_10"])
    ndcg = float(row["ndcg_at_10"])
    """前 5 就有明显不相关或弱相关结果时，优先指出前排质量问题。"""
    if p5 < 1.0:
        """返回分析文本。"""
        return "前 5 已出现不相关或弱相关结果，说明第一屏结果需要更严格过滤。"
    """前 10 不满时，指出尾部混入噪声。"""
    if p10 < 1.0:
        """返回分析文本。"""
        return "前 10 存在明显不相关结果，主要问题是尾部噪声混入。"
    """NDCG 偏低时，通常是强相关结果没有排到足够靠前。"""
    if ndcg < threshold:
        """返回分析文本。"""
        return "相关结果整体能召回，但强相关结果排序不够靠前。"
    """如果候选级仍有 bad case，则给出综合判断。"""
    if type_counts:
        """返回分析文本。"""
        return "整体指标尚可，但局部排序或弱相关前排仍可优化。"
    """兜底返回。"""
    return "未发现明显 query 级问题。"


def build_bad_queries(per_query: List[dict], bad_cases: List[dict], threshold: float) -> List[dict]:
    """
    生成 query 级 bad case 汇总。
    """
    """按 query_id 汇总候选级问题数量。"""
    cases_by_query: Dict[str, List[dict]] = {}
    """逐条 bad case 加入分组。"""
    for row in bad_cases:
        """读取 query_id。"""
        query_id = str(row.get("query_id"))
        """加入当前 query 的 bad case 列表。"""
        cases_by_query.setdefault(query_id, []).append(row)
    """准备 bad query 列表。"""
    bad_queries = []
    """逐个 query 判断是否需要进入重点分析。"""
    for row in per_query:
        """读取 query_id。"""
        query_id = str(row["query_id"])
        """读取当前 query 的候选级问题。"""
        query_cases = cases_by_query.get(query_id, [])
        """统计不同问题类型数量。"""
        type_counts = {
            "false_positive": sum(1 for item in query_cases if item["bad_case_type"] == "false_positive"),
            "weak_top_rank": sum(1 for item in query_cases if item["bad_case_type"] == "weak_top_rank"),
            "relevant_ranked_low": sum(1 for item in query_cases if item["bad_case_type"] == "relevant_ranked_low"),
        }
        """判断当前 query 是否进入重点 bad query。"""
        should_include = (
            float(row["precision_at_5"]) < 1.0
            or float(row["precision_at_10"]) < 1.0
            or float(row["ndcg_at_10"]) < threshold
        )
        """不需要分析则跳过。"""
        if not should_include:
            """继续下一条。"""
            continue
        """追加 query 级分析。"""
        bad_queries.append(
            {
                "query_id": query_id,
                "query": row["query"],
                "precision_at_5": row["precision_at_5"],
                "precision_at_10": row["precision_at_10"],
                "ndcg_at_5": row["ndcg_at_5"],
                "ndcg_at_10": row["ndcg_at_10"],
                "avg_latency_ms": row["avg_latency_ms"],
                "bad_case_count": len(query_cases),
                "false_positive_count": type_counts["false_positive"],
                "weak_top_rank_count": type_counts["weak_top_rank"],
                "relevant_ranked_low_count": type_counts["relevant_ranked_low"],
                "analysis": bad_query_analysis(row, type_counts, threshold),
            }
        )
    """把问题最明显的 query 排到前面。"""
    bad_queries.sort(key=lambda item: (float(item["ndcg_at_10"]), float(item["precision_at_10"]), item["query_id"]))
    """返回 bad query 汇总。"""
    return bad_queries


def command_score(args: argparse.Namespace) -> int:
    """
    读取人工标注结果并计算整体指标。
    """
    """读取带标签结果。"""
    rows = read_jsonl(args.labeled)
    """校验标签完整性。"""
    validate_labels(rows)
    """按 query 分组。"""
    grouped = group_by_query(rows)
    """准备逐 query 指标。"""
    per_query = []
    """逐组计算指标。"""
    for query_id, group in grouped.items():
        """读取 query 文本。"""
        query = str(group[0].get("query", ""))
        """计算平均耗时；同一 query 的 top10 记录相同，取平均也安全。"""
        latency_ms = mean([float(row.get("latency_ms", 0.0)) for row in group])
        """构造当前 query 指标。"""
        per_query.append(
            {
                "query_id": query_id,
                "query": query,
                "count": len(group),
                "precision_at_5": precision_at(group, 5),
                "precision_at_10": precision_at(group, 10),
                "ndcg_at_5": ndcg_at(group, 5),
                "ndcg_at_10": ndcg_at(group, 10),
                "avg_latency_ms": latency_ms,
            }
        )
    """按 query_id 排序，方便阅读。"""
    per_query.sort(key=lambda row: row["query_id"])
    """汇总整体指标。"""
    metrics = {
        "num_queries": len(per_query),
        "num_results": len(rows),
        "precision_at_5": mean([row["precision_at_5"] for row in per_query]),
        "precision_at_10": mean([row["precision_at_10"] for row in per_query]),
        "ndcg_at_5": mean([row["ndcg_at_5"] for row in per_query]),
        "ndcg_at_10": mean([row["ndcg_at_10"] for row in per_query]),
        "avg_latency_ms": mean([row["avg_latency_ms"] for row in per_query]),
        "p95_latency_ms": percentile([row["avg_latency_ms"] for row in per_query], 0.95),
        "latency_note": "avg_latency_ms 不包含模型、processor、FAISS 索引和 retrieval_meta 的初始化时间，只统计 pipeline 常驻后的单 query 搜索时间。",
    }
    """生成候选级 bad cases。"""
    bad_cases = build_bad_cases(rows)
    """生成 query 级 bad case 汇总。"""
    bad_queries = build_bad_queries(per_query, bad_cases, args.bad_ndcg_threshold)
    """把 bad case 数量补充进整体指标。"""
    metrics["num_bad_cases"] = len(bad_cases)
    """把 bad query 数量补充进整体指标。"""
    metrics["num_bad_queries"] = len(bad_queries)
    """写出整体指标。"""
    output_json = ensure_parent(args.output_json)
    """打开 json 输出。"""
    with output_json.open("w", encoding="utf-8") as f:
        """写出结构化指标。"""
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    """写出逐 query 指标。"""
    write_csv(args.output_csv, per_query, ["query_id", "query", "count", "precision_at_5", "precision_at_10", "ndcg_at_5", "ndcg_at_10", "avg_latency_ms"])
    """写出候选级 bad cases。"""
    write_jsonl(args.output_bad_cases_jsonl, bad_cases)
    """写出候选级 bad cases 表格。"""
    write_csv(args.output_bad_cases_csv, bad_cases, BAD_CASE_FIELDNAMES)
    """写出 query 级 bad cases 表格。"""
    write_csv(args.output_bad_queries_csv, bad_queries, BAD_QUERY_FIELDNAMES)
    """写出 bad cases 分析报告。"""
    write_bad_cases_summary(args.output_bad_cases_md, metrics, bad_queries, bad_cases)
    """写出摘要报告。"""
    write_summary(args.output_md, metrics, per_query, bad_queries)
    """打印核心指标。"""
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    """正常结束。"""
    return 0


def write_summary(path: str | Path, metrics: dict, per_query: List[dict], bad_queries: List[dict] | None = None) -> None:
    """
    写出 Markdown 摘要报告。
    """
    """确保输出目录存在。"""
    path = ensure_parent(path)
    """准备报告内容。"""
    lines = [
        "# 项目一人工评估摘要",
        "",
        "## 整体指标",
        "",
        f"- Query 数：{metrics['num_queries']}",
        f"- 结果数：{metrics['num_results']}",
        f"- Precision@5：{metrics['precision_at_5']:.4f}",
        f"- Precision@10：{metrics['precision_at_10']:.4f}",
        f"- NDCG@5：{metrics['ndcg_at_5']:.4f}",
        f"- NDCG@10：{metrics['ndcg_at_10']:.4f}",
        f"- 平均检索耗时：{metrics['avg_latency_ms']:.2f} ms",
        f"- P95 检索耗时：{metrics['p95_latency_ms']:.2f} ms",
        f"- Bad Cases 数：{metrics.get('num_bad_cases', 0)}",
        f"- Bad Queries 数：{metrics.get('num_bad_queries', 0)}",
        "",
        "说明：平均检索耗时不包含模型、processor、FAISS 索引和 retrieval_meta 的初始化时间，只统计 pipeline 常驻后的单 query 搜索时间。",
        "",
        "## 逐 Query 指标",
        "",
        "| query_id | query | P@5 | P@10 | NDCG@5 | NDCG@10 | latency_ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    """逐 query 追加表格行。"""
    for row in per_query:
        """追加一行指标。"""
        lines.append(
            f"| {row['query_id']} | {row['query']} | {row['precision_at_5']:.4f} | "
            f"{row['precision_at_10']:.4f} | {row['ndcg_at_5']:.4f} | "
            f"{row['ndcg_at_10']:.4f} | {row['avg_latency_ms']:.2f} |"
        )
    """如果有 bad query，则追加简短索引。"""
    if bad_queries:
        """追加分节标题。"""
        lines.extend(
            [
                "",
                "## Bad Cases 简析",
                "",
                "详细文件：`reports/manual_eval_bad_cases.md`",
                "",
                "| query_id | query | P@10 | NDCG@10 | 问题概述 |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        """只在主摘要中展示前十个最明显问题。"""
        for row in bad_queries[:10]:
            """追加一行 bad query 摘要。"""
            lines.append(
                f"| {row['query_id']} | {row['query']} | {row['precision_at_10']:.4f} | "
                f"{row['ndcg_at_10']:.4f} | {row['analysis']} |"
            )
    """写出报告。"""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bad_cases_summary(path: str | Path, metrics: dict, bad_queries: List[dict], bad_cases: List[dict]) -> None:
    """
    写出 bad cases 专项分析报告。
    """
    """确保输出目录存在。"""
    path = ensure_parent(path)
    """统计问题类型。"""
    false_positive_count = sum(1 for row in bad_cases if row["bad_case_type"] == "false_positive")
    """统计问题类型。"""
    weak_top_rank_count = sum(1 for row in bad_cases if row["bad_case_type"] == "weak_top_rank")
    """统计问题类型。"""
    relevant_ranked_low_count = sum(1 for row in bad_cases if row["bad_case_type"] == "relevant_ranked_low")
    """按 query 汇总 bad cases。"""
    cases_by_query: Dict[str, List[dict]] = {}
    """逐条加入分组。"""
    for row in bad_cases:
        """加入当前 query。"""
        cases_by_query.setdefault(str(row.get("query_id")), []).append(row)
    """准备报告内容。"""
    lines = [
        "# 项目一 Bad Cases 分析",
        "",
        "## 筛选口径",
        "",
        "- `false_positive`：人工标签为 0，说明明显不相关但进入 top10。",
        "- `weak_top_rank`：人工标签为 1 且排在前 5，说明第一屏有弱相关结果。",
        "- `relevant_ranked_low`：人工标签为 2 但排在第 6 名之后，说明强相关结果没有足够靠前。",
        "",
        "## 总览",
        "",
        f"- Query 数：{metrics['num_queries']}",
        f"- 结果数：{metrics['num_results']}",
        f"- 候选级 Bad Cases：{len(bad_cases)}",
        f"- Query 级 Bad Queries：{len(bad_queries)}",
        f"- false_positive：{false_positive_count}",
        f"- weak_top_rank：{weak_top_rank_count}",
        f"- relevant_ranked_low：{relevant_ranked_low_count}",
        "",
        "## 重点 Bad Queries",
        "",
        "| query_id | query | P@5 | P@10 | NDCG@10 | 问题数 | 分析 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    """追加 query 级问题表格。"""
    for row in bad_queries:
        """追加一行。"""
        lines.append(
            f"| {row['query_id']} | {row['query']} | {row['precision_at_5']:.4f} | "
            f"{row['precision_at_10']:.4f} | {row['ndcg_at_10']:.4f} | {row['bad_case_count']} | {row['analysis']} |"
        )
    """追加候选级问题标题。"""
    lines.extend(
        [
            "",
            "## 候选级 Bad Cases",
            "",
            "| query_id | query | rank | label | type | text | reason |",
            "| --- | --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    """只展示重点 bad query 下的候选级问题，避免 Markdown 过长。"""
    for query_row in bad_queries:
        """读取当前 query 的问题候选。"""
        query_cases = cases_by_query.get(query_row["query_id"], [])
        """按 rank 排序。"""
        query_cases.sort(key=lambda item: int(item.get("rank", 0)))
        """逐条追加。"""
        for case in query_cases:
            """追加一行候选级问题。"""
            lines.append(
                f"| {case['query_id']} | {case['query']} | {case['rank']} | {case['label']} | "
                f"{case['bad_case_type']} | {case['text']} | {case['bad_case_reason']} |"
            )
    """追加改进建议。"""
    lines.extend(
        [
            "",
            "## 改进方向",
            "",
            "- 对商品短 query，增加关键属性约束，避免只匹配宽泛类别。",
            "- 对场景类 query，继续保留图文双索引互补，但需要观察图像分是否把视觉细节提前。",
            "- 对强相关结果排在后面的 query，可优先调试 late fusion 中文本索引分和图片索引分的贡献。",
        ]
    )
    """写出报告。"""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """
    脚本入口。
    """
    """读取参数。"""
    args = parse_args()
    """按子命令分发。"""
    if args.command == "collect":
        """执行候选采集。"""
        return command_collect(args)
    """执行指标计算。"""
    if args.command == "score":
        """计算指标。"""
        return command_score(args)
    """未知命令直接报错。"""
    raise ValueError(args.command)


"""直接运行文件时执行入口。"""
if __name__ == "__main__":
    """把主函数返回值作为进程退出码。"""
    sys.exit(main())
