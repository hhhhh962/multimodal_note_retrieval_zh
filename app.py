"""
项目一 Gradio 检索 Demo。

页面支持文本查询、图片查询或图文联合查询。服务启动时加载 Chinese-CLIP、FAISS 索引和元信息，后续
每次点击搜索只做 query 编码和索引检索，避免命令行冷启动式的重复加载。
"""

"""引入系统环境工具，用来读取模型路径和端口配置。"""
import os
"""引入路径对象，用来拼接项目内图片路径。"""
from pathlib import Path
"""引入表格工具，用来展示检索结果。"""
import pandas as pd
"""引入 Gradio，用来构建可交互 Demo。"""
import gradio as gr

"""引入四路召回搜索流水线。"""
from src.search_pipeline import DEFAULT_EMBEDDING_DIR, DEFAULT_MODEL_DIR, SearchPipeline

"""记录项目根目录。"""
PROJECT_ROOT = Path(__file__).resolve().parent
"""记录索引目录，默认使用微调模型生成的全量索引。"""
EMBEDDING_DIR = os.environ.get("EMBEDDING_DIR", DEFAULT_EMBEDDING_DIR)
"""记录模型路径，默认使用已经合并 LoRA 的微调模型。"""
MODEL_NAME = (
    os.environ.get("MODEL_DIR")
    or os.environ.get("MODEL_NAME")
    or str(PROJECT_ROOT / DEFAULT_MODEL_DIR)
)
"""记录查询编码设备；auto 会在有显卡时优先使用 CUDA。"""
DEVICE = os.environ.get("DEVICE", "auto")
"""记录是否尝试使用显卡 FAISS；默认面向后续有卡运行。"""
USE_GPU_FAISS = os.environ.get("USE_GPU_FAISS", "1") not in {"0", "false", "False"}
"""可选覆盖 HNSW 查询深度；默认使用索引构建时选出的最高质量参数。"""
HNSW_EF_SEARCH_RAW = os.environ.get("HNSW_EF_SEARCH")
HNSW_EF_SEARCH = int(HNSW_EF_SEARCH_RAW) if HNSW_EF_SEARCH_RAW else None
"""记录固定 late fusion 系数。"""
ALPHA = float(os.environ.get("ALPHA", "0.5"))
"""记录每一路召回数量。"""
TOP_K_RECALL = int(os.environ.get("TOP_K_RECALL", "100"))
"""记录最终展示数量。"""
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "10"))

"""启动时创建并持有搜索流水线。"""
PIPELINE = SearchPipeline(
    embedding_dir=PROJECT_ROOT / EMBEDDING_DIR,
    model_name=MODEL_NAME,
    project_root=PROJECT_ROOT,
    device=DEVICE,
    use_gpu_faiss=USE_GPU_FAISS,
    ef_search=HNSW_EF_SEARCH,
)


def resolve_image_path(image_path: str | None) -> str | None:
    """Resolve one image from a document's complete image list."""

    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path) if path.exists() else None


def rows_to_table(results: list[dict]) -> pd.DataFrame:
    """
    把文档级搜索结果转成表格。
    """
    """准备表格行。"""
    rows = []
    """逐条整理展示字段。"""
    for rank, row in enumerate(results, start=1):
        """追加一行展示数据。"""
        image_paths = row.get("image_paths", [])
        rows.append(
            {
                "排名": rank,
                "综合分": round(float(row.get("score_mm", 0.0)), 4),
                "文本索引分": round(float(row.get("score_text_index", 0.0)), 4),
                "图片索引分": round(float(row.get("score_image_index", 0.0)), 4),
                "文搜文": round(float(row.get("score_tt", 0.0)), 4),
                "文搜图": round(float(row.get("score_ti", 0.0)), 4),
                "图片数": row.get("image_count", len(row.get("image_paths", []))),
                "文本": row.get("text"),
                "最佳匹配图": row.get("matched_image_path"),
                "来源": row.get("source"),
                "划分": row.get("split"),
                "编号": row.get("doc_id"),
            }
        )
    """返回表格对象。"""
    return pd.DataFrame(rows)


def search(query: str, image_path: str | None):
    """
    响应页面搜索按钮。
    """
    """执行四路召回搜索。"""
    try:
        """调用常驻流水线获取结果。"""
        results = PIPELINE.search(
            query=query,
            image_path=image_path,
            alpha=ALPHA,
            top_k_recall=TOP_K_RECALL,
            top_k_final=TOP_K_FINAL,
        )
    except Exception as exc:
        """出错时返回空结果和错误提示表格。"""
        return [], pd.DataFrame([{"错误": str(exc)}])
    """准备图片展示数据。"""
    gallery = []
    """按笔记排名连续展示每篇笔记的完整图片集合。"""
    for rank, row in enumerate(results, start=1):
        image_paths = list(row.get("image_paths") or [])
        for image_number, image_path in enumerate(image_paths, start=1):
            image = resolve_image_path(image_path)
            if image:
                gallery.append(
                    (
                        image,
                        f"笔记 {rank} · 图 {image_number}/{len(image_paths)} · "
                        f"{row.get('score_mm', 0.0):.4f} · {row.get('text')}",
                    )
                )
    """返回图片墙和表格。"""
    return gallery, rows_to_table(results)


"""创建 Gradio 页面。"""
with gr.Blocks(title="项目一中文图文搜索") as demo:
    """页面标题。"""
    gr.Markdown("# 项目一中文图文搜索")
    """输入区域。"""
    with gr.Row():
        """文本查询输入。"""
        query_input = gr.Textbox(label="文本查询", placeholder="例如：圣诞节沙发靠垫")
        """图片查询输入。"""
        image_input = gr.Image(label="图片查询", type="filepath")
    """搜索按钮。"""
    search_button = gr.Button("搜索", variant="primary")
    """图片结果区域。"""
    gallery_output = gr.Gallery(label="Top10 笔记的完整图片", columns=5, height=520)
    """表格结果区域。"""
    table_output = gr.Dataframe(label="Top10 结果明细", wrap=True)
    """绑定点击事件。"""
    search_button.click(search, inputs=[query_input, image_input], outputs=[gallery_output, table_output])


"""只有直接运行 app.py 时才启动服务。"""
if __name__ == "__main__":
    """读取服务端口。"""
    port = int(os.environ.get("PORT", "7860"))
    """启动 Gradio 服务。"""
    demo.launch(server_name="0.0.0.0", server_port=port)
