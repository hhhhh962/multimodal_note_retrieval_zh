#!/usr/bin/env python3
"""生成 MUGE 一文多图聚合映射（流式读取 items.jsonl，低内存）。"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--items",
    type=Path,
    default=PROJECT_ROOT / "data/processed/demo/items.jsonl",
)
parser.add_argument(
    "--output",
    type=Path,
    default=PROJECT_ROOT / "outputs/finetuned_full/muge_mapping.json",
)
args = parser.parse_args()
ITEMS_JSONL = args.items
OUTPUT = args.output

print(f"流式读取: {ITEMS_JSONL}")
text_to_rows = defaultdict(list)
row_to_text_id = {}
total = 0
muge_count = 0

with ITEMS_JSONL.open("r", encoding="utf-8") as f:
    for row_id, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        total += 1
        if item.get("source") != "MUGE":
            continue
        muge_count += 1
        text_id = item.get("text_id")
        if text_id is None:
            parts = item["item_id"].split("_")
            text_id = parts[2] if len(parts) >= 4 else item["item_id"]
        doc_id = f"MUGE_{item.get('split', 'unknown')}_{text_id}"
        text_to_rows[doc_id].append(row_id)
        row_to_text_id[str(row_id)] = doc_id

print(f"总 item 数: {total}")
print(f"MUGE item 数: {muge_count}")
print(f"MUGE 文档数(聚合后): {len(text_to_rows)}")

multi_image = sum(1 for rows in text_to_rows.values() if len(rows) > 1)
print(f"多图文档数(>1张图): {multi_image}")
max_images = max(len(rows) for rows in text_to_rows.values()) if text_to_rows else 0
print(f"单文档最多图片数: {max_images}")

output = {
    "text_to_rows": dict(text_to_rows),
    "row_to_text_id": row_to_text_id,
    "stats": {
        "total_items": total,
        "muge_items": muge_count,
        "muge_documents": len(text_to_rows),
        "multi_image_documents": multi_image,
        "max_images_per_doc": max_images,
    },
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
print(f"已保存: {OUTPUT}")
print(f"文件大小: {OUTPUT.stat().st_size} bytes")
