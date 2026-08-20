"""
命令行检索入口。

这个文件只负责读取命令行参数、调用四路召回搜索流水线、再把结果打印出来。核心检索逻辑统一放在
search_pipeline.py，避免 Demo 和命令行出现两套不一致的实现。
"""

"""引入命令行参数解析工具。"""
import argparse
"""引入结构化文本工具，用来按需输出 JSON 结果。"""
import json
"""引入系统退出工具，用来返回进程状态码。"""
import sys
"""引入类型标注，说明结果列表内容。"""
from typing import List

"""引入四路召回搜索流水线。"""
from search_pipeline import DEFAULT_EMBEDDING_DIR, SearchPipeline


def parse_args() -> argparse.Namespace:
    """
    读取检索参数。

    query 和 image_path 都是可选参数，但运行时至少需要提供一个。
    """
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="使用 Chinese-CLIP 和 text/image FAISS 做四路召回检索")
    """声明索引和元信息所在目录。"""
    parser.add_argument("--embedding_dir", default=DEFAULT_EMBEDDING_DIR)
    """声明项目根目录，用来解析相对图片路径。"""
    parser.add_argument("--project_root", default=".")
    """声明查询编码使用的模型名称或本地模型目录。"""
    parser.add_argument("--model_name", default=None)
    """声明可选文本查询。"""
    parser.add_argument("--query", default=None)
    """声明可选图片查询路径。"""
    parser.add_argument("--image_path", default=None)
    """声明文本索引分数和图片索引分数的固定融合系数。"""
    parser.add_argument("--alpha", type=float, default=0.5)
    """声明每一路先召回多少候选。"""
    parser.add_argument("--top_k_recall", type=int, default=100)
    """声明最终输出多少条结果。"""
    parser.add_argument("--top_k_final", type=int, default=10)
    """声明查询编码运行设备；auto 会优先使用 CUDA。"""
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    """声明是否在显卡上使用半精度查询编码。"""
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    """声明是否输出结构化 JSON。"""
    parser.add_argument("--json", action="store_true")
    """解析并返回参数。"""
    return parser.parse_args()


def print_text_results(results: List[dict]) -> None:
    """
    按适合终端阅读的格式打印搜索结果（文档级，MUGE 多图聚合）。
    """
    """逐条输出最终结果。"""
    for rank, row in enumerate(results, start=1):
        """打印排名、综合分、两类索引分和类型。"""
        doc_type = row.get("type", "single")
        item_count = row.get("item_count", 1)
        type_label = f"聚合({item_count}图)" if doc_type == "aggregated" else "独立"
        print(
            f"排名={rank:02d} 综合分={row['score_mm']:.4f} "
            f"文本索引分={row['score_text_index']:.4f} 图片索引分={row['score_image_index']:.4f} "
            f"tt={row['score_tt']:.4f} ti={row['score_ti']:.4f} "
            f"[{type_label}] "
            f"编号={row.get('doc_id')} 来源={row.get('source')} 划分={row.get('split')}"
        )
        """打印候选文本。"""
        print(f"    文本: {row.get('text')}")
        """打印候选图片路径（MUGE 聚合可能有多张）。"""
        image_paths = row.get("image_paths", [])
        if len(image_paths) == 1:
            print(f"    图片: {image_paths[0]}")
        else:
            print(f"    图片({len(image_paths)}张): {', '.join(image_paths[:3])}{'...' if len(image_paths) > 3 else ''}")


def main() -> int:
    """
    执行命令行检索。
    """
    """读取命令行参数。"""
    args = parse_args()
    """在加载模型和索引前先检查至少存在一种查询输入。"""
    if not ((args.query or "").strip() or (args.image_path or "").strip()):
        """缺少查询输入时直接给出清晰错误，避免无意义冷启动。"""
        raise SystemExit("请至少提供 --query 或 --image_path 中的一个")
    """创建四路召回搜索流水线。"""
    pipeline = SearchPipeline(
        embedding_dir=args.embedding_dir,
        model_name=args.model_name,
        project_root=args.project_root,
        device=args.device,
        fp16=args.fp16,
    )
    """执行搜索。"""
    results = pipeline.search(
        query=args.query,
        image_path=args.image_path,
        alpha=args.alpha,
        top_k_recall=args.top_k_recall,
        top_k_final=args.top_k_final,
    )
    """按用户选择输出 JSON 或终端文本。"""
    if args.json:
        """打印结构化结果，方便后续评估脚本读取。"""
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        """打印便于人工查看的结果。"""
        print_text_results(results)
    """正常结束。"""
    return 0


"""只有当本文件作为脚本直接运行时，才执行主入口。"""
if __name__ == "__main__":
    """把主函数返回值作为进程退出码。"""
    sys.exit(main())
