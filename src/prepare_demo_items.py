"""
演示数据准备脚本。

脚本从已经预处理好的图文数据集中抽取可用于项目一演示的样本，生成统一的样本清单、
图片文件、人工查询集、评估模板和统计摘要。
"""

"""引入命令行参数解析工具，用来读取运行脚本时传入的路径和配置。"""
import argparse
"""引入图片解码工具，用来把原始表格里的图片编码还原成图片二进制。"""
import base64
"""引入结构化文本工具，用来读取和写入一行一条记录的数据。"""
import json
"""引入随机数工具，用于抽样检查图片时固定抽样结果。"""
import random
"""引入文件操作工具，用来删除目录和查看磁盘空间。"""
import shutil
"""引入路径对象，方便用斜杠拼接目录和文件名。"""
from pathlib import Path

"""引入图片读取库，用于检查解码后的图片文件是否能正常打开。"""
from PIL import Image


"""定义需要处理的数据集，以及每个数据集要处理哪些划分。"""
SOURCE_SPLITS = {
    "MUGE": ["train", "valid"],
    "Flickr30k-CN": ["train", "valid", "test"],
    "COCO-CN": ["train", "valid", "test"],
}


"""定义第一版人工评估查询集；每个元组包含查询编号和查询文本。"""
DEFAULT_MANUAL_QUERIES = [
    ("manual_001", "圣诞节沙发靠垫"),
    ("manual_002", "红色家居装饰"),
    ("manual_003", "手机壳粘土"),
    ("manual_004", "男士长袖纯色上衣"),
    ("manual_005", "透明亚克力展示盒"),
    ("manual_006", "可爱的黄色小狗"),
    ("manual_007", "狗在水里游泳"),
    ("manual_008", "飞机在蓝天飞行"),
    ("manual_009", "复古蒸汽火车"),
    ("manual_010", "小孩穿条纹裤"),
    ("manual_011", "婴儿穿粉色衣服躺着"),
    ("manual_012", "警犬坐在警察旁边"),
    ("manual_013", "草地上的几只狗"),
    ("manual_014", "干净整洁的房间"),
    ("manual_015", "早餐盘子和刀叉"),
    ("manual_016", "街边公共卫生间"),
    ("manual_017", "黑白照片里的长椅"),
    ("manual_018", "户外露营准备"),
    ("manual_019", "蓝天上的大型飞机"),
    ("manual_020", "家居花瓶摆件"),
    ("manual_021", "黑金男士戒指"),
    ("manual_022", "木质床头柜"),
    ("manual_023", "儿童玩具车"),
    ("manual_024", "长椅上的黄色小狗"),
    ("manual_025", "厨房餐具和食物"),
    ("manual_026", "多人户外活动"),
    ("manual_027", "城市街道临时设施"),
    ("manual_028", "商品包装展示图"),
    ("manual_029", "休闲男装穿搭"),
    ("manual_030", "温馨卧室布置"),
]


def reset_dir(path):
    """
    重建目录。

    如果目录已经存在，先整体删除再重新创建，保证本次生成结果不受旧文件影响。
    """
    """判断目标目录是否已经存在。"""
    if path.exists():
        """目录存在时递归删除目录和其中的所有文件，避免旧结果混入本次输出。"""
        shutil.rmtree(path)
    """重新创建目标目录；如果上级目录不存在，也会一起创建。"""
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path, rows):
    """
    写入“一行一条记录”的结构化文本文件。

    每条记录单独占一行，便于后续流式读取和追加处理。
    """
    """先确保输出文件所在的目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    """用写入模式打开目标文件，并指定中文友好的编码。"""
    with path.open("w", encoding="utf-8") as f:
        """逐条遍历准备写入的记录。"""
        for row in rows:
            """把一条字典记录转成一行文本并写入；中文保持原样，不转成转义形式。"""
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_text_items(texts_path, source, split):
    """
    从文本清单中生成样本行。

    一条文本可能关联多张图片，因此会展开成多条“文本加图片”的样本；同时收集需要解码的
    图片编号，供后续只解码真正用到的图片。
    """
    """记录原始文本文件中读到的非空行数量。"""
    text_rows = 0
    """记录同时包含文本和图片编号的可用文本行数量。"""
    usable_text_rows = 0
    """保存展开后的样本行；每一行对应一条文本加一张图片。"""
    item_rows = []
    """保存后续必须解码的图片编号；集合可以自动去重。"""
    needed_ids = set()

    """打开当前数据集划分对应的文本清单。"""
    with texts_path.open("r", encoding="utf-8") as f:
        """逐行读取文本清单，因为原始文件是一行一条记录。"""
        for line in f:
            """跳过空行，避免空内容参与统计和解析。"""
            if not line.strip():
                """进入下一行，不再处理当前空行。"""
                continue
            """非空行计入原始文本行数量。"""
            text_rows += 1
            """把当前行从文本形式解析成字典。"""
            obj = json.loads(line)
            """取出文本字段；如果字段缺失则按空字符串处理，并去掉首尾空白。"""
            text = str(obj.get("text", "")).strip()
            """取出图片编号列表，并统一转成字符串，同时过滤空编号。"""
            image_ids = [str(x) for x in obj.get("image_ids", []) if str(x)]
            """如果文本为空或没有任何图片编号，这条记录不能组成图文样本。"""
            if not text or not image_ids:
                """跳过不可用记录。"""
                continue

            """走到这里说明当前文本记录可用，可用计数加一。"""
            usable_text_rows += 1
            """取出原始文本编号，后续用于构造样本编号和追踪来源。"""
            text_id = obj.get("text_id")
            """一条文本可能对应多张图片，所以按图片编号逐个展开。"""
            for image_id in image_ids:
                """记录这张图片后续需要从原始图片表中解码出来。"""
                needed_ids.add(image_id)
                """追加一条统一格式的样本行。"""
                item_rows.append(
                    {
                        "item_id": f"{source}_{split}_{text_id}_{image_id}",
                        "text": text,
                        "image_id": image_id,
                        "image_path": f"data/images/demo/{source}/{split}/{image_id}.jpg",
                        "source": source,
                        "split": split,
                        "text_id": text_id,
                    }
                )

    """返回统计数量、展开后的样本行，以及需要解码的图片编号集合。"""
    return text_rows, usable_text_rows, item_rows, needed_ids


def decode_needed_images(imgs_path, image_out_dir, needed_ids):
    """
    解码并保存需要的图片。

    原始图片文件中包含大量图片；这里只处理文本样本实际引用到的图片，减少磁盘占用和处理时间。
    """
    """保存已经成功解码的图片编号。"""
    decoded_ids = set()
    """保存原始图片表中重复出现的图片编号。"""
    duplicate_ids = []
    """保存格式错误、解码失败或写入失败的图片记录。"""
    decode_failed = []
    """记录原始图片表中总共扫描了多少行。"""
    image_rows_seen = 0

    """确保当前划分的图片输出目录存在。"""
    image_out_dir.mkdir(parents=True, exist_ok=True)
    """打开原始图片表；遇到无法解码的字符时替换掉，避免读取阶段直接崩溃。"""
    with imgs_path.open("r", encoding="utf-8", errors="replace") as f:
        """逐行扫描原始图片表。"""
        for line in f:
            """每读到一行，扫描行数加一。"""
            image_rows_seen += 1
            """按第一个制表符拆分图片编号和图片编码内容。"""
            parts = line.rstrip("\n").split("\t", 1)
            """正常行应该刚好包含图片编号和编码内容两部分。"""
            if len(parts) != 2:
                """格式不对时记录失败原因，并尽量保留当前行里能看到的图片编号。"""
                decode_failed.append({"image_id": parts[0] if parts else "", "reason": "bad_tsv_columns"})
                """格式错误的行不能继续解码，直接处理下一行。"""
                continue

            """把拆出来的两部分分别命名为图片编号和编码内容。"""
            image_id, encoded = parts
            """如果这张图片没有被任何样本引用，就不需要解码保存。"""
            if image_id not in needed_ids:
                """跳过不需要的图片，节省时间和磁盘空间。"""
                continue
            """如果图片之前已经成功解码过，说明原始图片表里有重复编号。"""
            if image_id in decoded_ids:
                """记录重复编号，便于后续统计数据质量。"""
                duplicate_ids.append(image_id)
                """重复图片不再重复写入。"""
                continue

            """开始尝试把图片编码还原成图片文件。"""
            try:
                """把文本形式的图片编码还原成二进制图片内容。"""
                image_bytes = base64.b64decode(encoded)
                """把二进制内容写成图片文件，文件名使用图片编号。"""
                (image_out_dir / f"{image_id}.jpg").write_bytes(image_bytes)
                """写入成功后，把图片编号加入成功集合。"""
                decoded_ids.add(image_id)
            except Exception as exc:
                """如果解码或写入失败，记录图片编号和具体异常信息。"""
                decode_failed.append({"image_id": image_id, "reason": repr(exc)})

            """每成功解码一万张图片，就输出一次进度。"""
            if len(decoded_ids) and len(decoded_ids) % 10000 == 0:
                """立刻刷新输出，方便远程日志及时看到进度。"""
                print(f"已解码图片数：{len(decoded_ids)}，来源文件：{imgs_path.name}", flush=True)

    """返回扫描行数、成功图片编号、重复图片编号和失败记录。"""
    return image_rows_seen, decoded_ids, duplicate_ids, decode_failed


def check_images(project_root, item_paths, sample_size, seed):
    """
    抽样检查图片是否能被正常打开。

    不对全部图片逐张校验，是为了节省准备阶段时间；抽样失败会直接报错，提示数据质量问题。
    """
    """创建带固定种子的随机数生成器，保证每次抽样可复现。"""
    rng = random.Random(seed)
    """对样本图片路径去重并排序，避免重复图片影响抽样。"""
    unique_paths = sorted(set(item_paths))
    """如果图片总数不超过抽样上限，就全部检查；否则按固定随机种子抽样。"""
    sample = unique_paths if len(unique_paths) <= sample_size else rng.sample(unique_paths, sample_size)
    """保存打不开或校验失败的图片记录。"""
    failures = []
    """逐张检查抽样图片。"""
    for rel_path in sample:
        """把相对路径拼接到项目根目录下面，得到真实图片路径。"""
        abs_path = project_root / rel_path
        """尝试打开并校验图片文件。"""
        try:
            """打开图片文件；结束后自动关闭文件句柄。"""
            with Image.open(abs_path) as img:
                """校验图片格式和文件完整性。"""
                img.verify()
        except Exception as exc:
            """如果打开或校验失败，记录相对路径和异常信息。"""
            failures.append({"image_path": rel_path, "reason": repr(exc)})
    """返回本次检查的图片数量和失败列表。"""
    return len(sample), failures


def write_manual_queries(project_root, output_name):
    """
    写入第一版人工查询集。

    这些查询用于人工观察检索结果，也可作为后续计算人工评估指标的基础。
    """
    """把默认查询列表转换成一行一条记录的字典列表。"""
    rows = [
        {
            "query_id": query_id,
            "query": query,
            "label_rule": "2=高度相关，1=部分相关，0=不相关",
        }
        for query_id, query in DEFAULT_MANUAL_QUERIES
    ]
    """拼出人工查询集的输出路径。"""
    path = project_root / "data" / "processed" / output_name / "manual_queries.jsonl"
    """把人工查询集写成一行一条记录的文件。"""
    write_jsonl(path, rows)
    """返回写出的文件路径，供摘要文件记录。"""
    return path


def write_eval_template(project_root):
    """
    写入人工评估模板。

    这个模板给后续人工标注使用，记录每个查询的结果排序、相关性标签和备注。
    """
    """准备人工评估模板正文。"""
    template = """演示路线人工评估模板
====================

请使用 `data/processed/demo/manual_queries.jsonl` 作为第一版人工查询集。

标注规则
--------

```text
2 = 高度相关
1 = 部分相关
0 = 不相关
```

评估指标
--------

```text
前 5 条准确率
前 10 条准确率
前 10 条归一化折损累计增益
平均相关性分数
```

标注表
------

| 查询编号 | 查询词 | 排名 | 样本编号 | 来源 | 划分 | 分数 | 标签 | 备注 |
|---|---|---:|---|---|---|---:|---:|---|
| manual_001 | 圣诞节沙发靠垫 | 1 |  |  |  |  |  |  |

排序方案记录
------------

```text
不做多套手工调参方案。
当前只记录一个工程基线排序结果，以及每条结果的文本分、图片分和综合分。
如果后续采用论文或开源项目中更有依据的图文融合方法，再在这里记录唯一采用的方案、来源和理由。
```
"""
    """拼出评估模板的输出路径。"""
    path = project_root / "reports" / "demo_eval_template.md"
    """确保报告目录存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    """把模板正文写入文件。"""
    path.write_text(template, encoding="utf-8")
    """返回模板路径，供摘要文件记录。"""
    return path


def main():
    """
    主入口。

    按数据集和划分逐批生成演示样本，最后写出清单、查询、模板和摘要。
    """
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="从预处理图文数据中准备演示样本")
    """声明原始数据集根目录参数；这是必填项。"""
    parser.add_argument("--datasets-root", type=Path, required=True)
    """声明项目根目录参数；这是必填项。"""
    parser.add_argument("--project-root", type=Path, required=True)
    """声明输出目录名称，默认写入演示目录。"""
    parser.add_argument("--output-name", default="demo")
    """声明图片抽样检查数量，默认抽查二十张。"""
    parser.add_argument("--pil-sample-size", type=int, default=20)
    """声明随机种子，用于保证抽样检查结果可复现。"""
    parser.add_argument("--seed", type=int, default=20260703)
    """解析命令行传入的实际参数。"""
    args = parser.parse_args()

    """把原始数据集根目录转换成绝对路径。"""
    datasets_root = args.datasets_root.resolve()
    """把项目根目录转换成绝对路径。"""
    project_root = args.project_root.resolve()
    """拼出图片输出根目录。"""
    image_root = project_root / "data" / "images" / args.output_name
    """拼出处理后数据输出目录。"""
    processed_root = project_root / "data" / "processed" / args.output_name
    """拼出最终样本清单路径。"""
    items_path = processed_root / "items.jsonl"

    """清空并重建图片输出目录。"""
    reset_dir(image_root)
    """清空并重建处理后数据输出目录。"""
    reset_dir(processed_root)

    """保存已经写入过的样本编号，用于检查重复样本。"""
    item_ids = set()
    """保存所有写入样本的图片路径，供后续抽样检查。"""
    item_paths_for_check = []
    """保存每个数据集和每个划分的处理统计。"""
    summary_sources = []
    """统计全局写入样本数量。"""
    total_items_written = 0
    """统计按划分累加的成功解码图片数量。"""
    total_unique_images = 0

    """打开最终样本清单文件，准备逐行写入样本。"""
    with items_path.open("w", encoding="utf-8") as items_file:
        """遍历每个数据集及其需要处理的数据划分。"""
        for source, splits in SOURCE_SPLITS.items():
            """为当前数据集创建统计摘要容器。"""
            source_summary = {"source": source, "splits": []}
            """遍历当前数据集的每一个划分。"""
            for split in splits:
                """拼出当前划分的文本清单路径。"""
                texts_path = datasets_root / source / f"{split}_texts.jsonl"
                """拼出当前划分的原始图片表路径。"""
                imgs_path = datasets_root / source / f"{split}_imgs.tsv"
                """如果文本清单或图片表缺失，就跳过当前划分。"""
                if not texts_path.exists() or not imgs_path.exists():
                    """继续处理下一个划分。"""
                    continue

                """输出当前正在处理的数据集和划分。"""
                print(f"正在处理：{source}/{split}", flush=True)
                """从文本清单生成样本行，并收集需要解码的图片编号。"""
                text_rows, usable_text_rows, item_rows, needed_ids = iter_text_items(texts_path, source, split)
                """如果当前划分没有任何可用样本，就只记录统计并跳过。"""
                if not item_rows:
                    """把当前划分的空结果写入统计摘要。"""
                    source_summary["splits"].append(
                        {
                            "split": split,
                            "text_rows": text_rows,
                            "usable_text_rows": usable_text_rows,
                            "raw_item_count": 0,
                            "needed_image_count": 0,
                            "written_item_count": 0,
                            "decoded_image_count": 0,
                            "skipped_reason": "没有可用图片编号",
                        }
                    )
                    """输出当前划分没有可用样本的提示。"""
                    print(f"{source}/{split}：没有可用样本", flush=True)
                    """继续处理下一个划分。"""
                    continue

                """拼出当前划分的图片输出目录。"""
                image_out_dir = image_root / source / split
                """只解码当前样本实际需要的图片。"""
                image_rows_seen, decoded_ids, duplicate_ids, decode_failed = decode_needed_images(
                    imgs_path, image_out_dir, needed_ids
                )
                """找出需要但没有成功解码的图片编号。"""
                missing_ids = sorted(needed_ids - decoded_ids)
                """如果存在缺失图片，直接报错，避免后续生成引用坏路径的样本。"""
                if missing_ids:
                    """抛出缺失图片错误，并展示前几个样例。"""
                    raise RuntimeError(f"{source}/{split}：存在缺失图片，样例={missing_ids[:10]}")
                """如果存在解码失败记录，直接报错，提示数据质量或解码问题。"""
                if decode_failed:
                    """抛出解码失败错误，并展示前几个样例。"""
                    raise RuntimeError(f"{source}/{split}：存在图片解码失败，样例={decode_failed[:5]}")

                """记录当前划分实际写入了多少条样本。"""
                written = 0
                """遍历当前划分已经展开好的样本行。"""
                for item in item_rows:
                    """检查样本编号是否已经写入过。"""
                    if item["item_id"] in item_ids:
                        """如果样本编号重复，说明数据或构造规则有问题，直接报错。"""
                        raise RuntimeError(f"重复样本编号：{item['item_id']}")
                    """如果样本引用的图片没有成功解码，就跳过这条样本。"""
                    if item["image_id"] not in decoded_ids:
                        """继续处理下一条样本。"""
                        continue
                    """记录当前样本编号，后续可用于重复检测。"""
                    item_ids.add(item["item_id"])
                    """记录当前样本图片路径，后续用于抽样检查图片文件。"""
                    item_paths_for_check.append(item["image_path"])
                    """把当前样本写入最终样本清单，保持一行一条记录。"""
                    items_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                    """当前划分写入样本数加一。"""
                    written += 1

                """全局写入样本数累加当前划分结果。"""
                total_items_written += written
                """全局图片数累加当前划分成功解码的图片数。"""
                total_unique_images += len(decoded_ids)
                """整理当前划分的详细统计信息。"""
                split_summary = {
                    "split": split,
                    "text_rows": text_rows,
                    "usable_text_rows": usable_text_rows,
                    "raw_item_count": len(item_rows),
                    "needed_image_count": len(needed_ids),
                    "image_rows_seen": image_rows_seen,
                    "decoded_image_count": len(decoded_ids),
                    "written_item_count": written,
                    "duplicate_image_id_count": len(duplicate_ids),
                    "decode_failed_count": len(decode_failed),
                    "missing_image_id_count": len(missing_ids),
                    "duplicate_image_ids_sample": duplicate_ids[:20],
                }
                """把当前划分统计加入当前数据集摘要。"""
                source_summary["splits"].append(split_summary)
                """输出当前划分的处理结果。"""
                print(
                    f"{source}/{split}：可用文本={usable_text_rows}，样本={written}，图片={len(decoded_ids)}",
                    flush=True,
                )

            """当前数据集所有划分处理完后，把数据集摘要加入总摘要。"""
            summary_sources.append(source_summary)

    """抽样检查已写入样本引用的图片能否正常打开。"""
    checked_count, pil_failures = check_images(project_root, item_paths_for_check, args.pil_sample_size, args.seed)
    """如果抽样检查发现坏图，就直接报错。"""
    if pil_failures:
        """抛出图片校验失败错误，并展示前几个失败样例。"""
        raise RuntimeError(f"图片抽样校验失败，样例={pil_failures[:5]}")

    """写入第一版人工查询集，并记录输出路径。"""
    manual_queries_path = write_manual_queries(project_root, args.output_name)
    """写入人工评估模板，并记录输出路径。"""
    eval_template_path = write_eval_template(project_root)
    """读取项目所在磁盘的空间使用情况。"""
    disk_usage = shutil.disk_usage(project_root)
    """汇总本次数据准备的整体统计信息。"""
    summary = {
        "output_name": args.output_name,
        "items_path": str(items_path),
        "manual_queries_path": str(manual_queries_path),
        "eval_template_path": str(eval_template_path),
        "total_items_written": total_items_written,
        "total_unique_images_by_split_sum": total_unique_images,
        "unique_item_id_count": len(item_ids),
        "pil_sample_checked": checked_count,
        "pil_sample_failed_count": len(pil_failures),
        "disk_usage_bytes": {"total": disk_usage.total, "used": disk_usage.used, "free": disk_usage.free},
        "sources": summary_sources,
    }
    """拼出统计摘要文件路径。"""
    summary_path = processed_root / "summary.json"
    """把统计摘要写成格式化的结构化文本文件。"""
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    """输出最终写入的样本数量。"""
    print(f"样本数={total_items_written}", flush=True)
    """输出统计摘要文件路径。"""
    print(f"摘要路径={summary_path}", flush=True)


"""只有当本文件作为脚本直接运行时，才执行主入口。"""
if __name__ == "__main__":
    """调用主入口函数。"""
    main()
