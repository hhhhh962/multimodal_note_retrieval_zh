"""MUGE 一文多图聚合模块。

只对 source == 'MUGE' 的图文对按 text_id 聚合，同一 text_id 的多张图合并为一个文档。
Flickr30k-CN / COCO-CN 的一图多文保持独立样本，不聚合。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_muge_mapping(mapping_path: str | Path) -> dict[str, Any]:
    """加载预生成的 MUGE 聚合映射。

    返回:
        {
            "text_to_rows": {doc_id: [row_id, ...]},
            "row_to_text_id": {row_id_str: doc_id},
            "stats": {...}
        }
    """
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"MUGE 映射文件不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results(
    scored_rows: Sequence[Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    row_to_text_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    """把图文对搜索结果聚合为文档级结果。

    MUGE 按 text_id 合并（取组内最高 score_mm，图片路径合并为列表）；
    Flickr/COCO 保持独立（image_paths 为单元素列表）。

    Args:
        scored_rows: 已按 score_mm 降序排列的图文对结果，每个元素含 row_id, score_mm, score_tt, score_ti 等
        items: 完整的 item_meta 列表，items[row_id] 为该图文对元信息
        row_to_text_id: MUGE row_id(str) -> doc_id 映射

    Returns:
        聚合后的文档列表，按 score_mm 降序排列。每个元素含:
        - doc_id, type ("aggregated" | "single"), text, image_paths, best_image_path
        - score_mm, score_tt, score_ti, source, split, item_count
    """
    # 第一遍：按 doc_id 分组
    groups: dict[str, list[Mapping[str, Any]]] = {}
    singles: list[Mapping[str, Any]] = []

    for row in scored_rows:
        row_id = int(row["row_id"])
        item = items[row_id]
        source = item.get("source", "")

        if source == "MUGE":
            doc_id = row_to_text_id.get(str(row_id))
            if doc_id is None:
                # 映射里没有的 MUGE item，退化为独立样本
                singles.append(row)
                continue
            groups.setdefault(doc_id, []).append(row)
        else:
            singles.append(row)

    results: list[dict[str, Any]] = []

    # 处理 MUGE 聚合组
    for doc_id, group_rows in groups.items():
        # 组内按 score_mm 降序，取最高分的那条作为代表
        best = max(group_rows, key=lambda r: float(r.get("score_mm", 0.0)))
        best_row_id = int(best["row_id"])
        best_item = items[best_row_id]

        # 收集组内所有图片路径（按原始 row_id 顺序去重）
        image_paths: list[str] = []
        seen_images: set[str] = set()
        for r in sorted(group_rows, key=lambda x: int(x["row_id"])):
            r_item = items[int(r["row_id"])]
            img = r_item.get("image_path", "")
            if img and img not in seen_images:
                image_paths.append(img)
                seen_images.add(img)

        results.append({
            "doc_id": doc_id,
            "type": "aggregated",
            "text": best_item.get("text", ""),
            "image_paths": image_paths,
            "best_image_path": best_item.get("image_path", ""),
            "score_mm": float(best.get("score_mm", 0.0)),
            "score_tt": float(best.get("score_tt", 0.0)),
            "score_ti": float(best.get("score_ti", 0.0)),
            "score_text_index": float(best.get("score_text_index", 0.0)),
            "score_image_index": float(best.get("score_image_index", 0.0)),
            "source": "MUGE",
            "split": best_item.get("split", ""),
            "item_count": len(group_rows),
        })

    # 处理非 MUGE 独立样本（Flickr/COCO + 映射缺失的 MUGE）
    for row in singles:
        row_id = int(row["row_id"])
        item = items[row_id]
        source = item.get("source", "")
        doc_id = item.get("item_id", f"row_{row_id}")
        image_path = item.get("image_path", "")
        results.append({
            "doc_id": doc_id,
            "type": "single",
            "text": item.get("text", ""),
            "image_paths": [image_path] if image_path else [],
            "best_image_path": image_path,
            "score_mm": float(row.get("score_mm", 0.0)),
            "score_tt": float(row.get("score_tt", 0.0)),
            "score_ti": float(row.get("score_ti", 0.0)),
            "score_text_index": float(row.get("score_text_index", 0.0)),
            "score_image_index": float(row.get("score_image_index", 0.0)),
            "source": source,
            "split": item.get("split", ""),
            "item_count": 1,
        })

    # 聚合后重新按 score_mm 降序排列
    results.sort(key=lambda r: r["score_mm"], reverse=True)
    return results
