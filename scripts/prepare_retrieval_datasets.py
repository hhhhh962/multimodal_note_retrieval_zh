#!/usr/bin/env python3
"""检查并把 Flickr30k-CN、COCO-CN 原始数据转换为 Chinese-CLIP TSV/JSONL。"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image


SPLITS = ("train", "valid", "test")
EXPECTED = {
    "Flickr30k-CN": {
        "images": {"train": 29_783, "valid": 1_000, "test": 1_000},
        "texts_before_dedup": {"train": 148_915, "valid": 5_000, "test": 5_000},
        "texts_after_dedup": {"train": 148_673, "valid": 5_000, "test": 5_000},
    },
    "COCO-CN": {
        "images": {"train": 18_341, "valid": 1_000, "test": 1_000},
        "texts_before_dedup": {"train": 20_065, "valid": 1_100, "test": 1_053},
        "texts_after_dedup": {"train": 20_011, "valid": 1_100, "test": 1_053},
    },
}
TERMINAL_PUNCTUATION = set("。！？!?；;…….．")
COCO_ID_PATTERN = re.compile(r"^COCO_(train|val)2014_([0-9]{12})$")
POS_SUFFIX_PATTERN = re.compile(r":[A-Za-z]+$")


class DataError(RuntimeError):
    """表示源数据或转换结果不满足确定的数据契约。"""


@dataclass
class ImageEntry:
    source_id: str
    image_id: int
    path: Path
    sha256: str | None = None


@dataclass(frozen=True)
class TextEntry:
    source_text_id: str
    text_id: int
    image_id: int
    raw_text: str
    text: str


@dataclass
class SplitData:
    images: list[ImageEntry] = field(default_factory=list)
    texts: list[TextEntry] = field(default_factory=list)
    source_text_count: int = 0
    removed_duplicate_count: int = 0


@dataclass
class DatasetData:
    name: str
    splits: dict[str, SplitData]
    issues: "IssueTracker" = field(default_factory=lambda: IssueTracker())


@dataclass
class IssueTracker:
    counts: Counter = field(default_factory=Counter)
    samples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add(self, reason: str, message: str) -> None:
        self.counts[reason] += 1
        if len(self.samples[reason]) < 5:
            self.samples[reason].append(message)

    def print_summary(self, dataset_name: str) -> None:
        if not self.counts:
            print(f"[{dataset_name}] 跳过/警告记录：0")
            return
        print(f"[{dataset_name}] 跳过/警告汇总：")
        for reason, count in sorted(self.counts.items()):
            print(f"  {reason}: {count}")
            for sample in self.samples[reason]:
                print(f"    - {sample}")


def strict_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise DataError(f"文件不是合法 UTF-8：{path}: {exc}") from exc


def remove_control_characters(text: str) -> str:
    return "".join(character for character in text if not unicodedata.category(character).startswith("C"))


def normalize_common(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = remove_control_characters(text)
    return re.sub(r"\s+", " ", text).strip()


def ensure_terminal_punctuation(text: str) -> str:
    if text and text[-1] not in TERMINAL_PUNCTUATION:
        return text + "。"
    return text


def clean_flickr_machine_text(text: str) -> str:
    """还原 zhb 文本：单空格是分词边界，连续空格是中文逗号边界。"""
    text = remove_control_characters(unicodedata.normalize("NFKC", text)).strip()
    chunks = []
    for chunk in re.split(r"\s{2,}", text):
        chunk = re.sub(r"\s+", "", chunk)
        if chunk:
            chunks.append(chunk)
    return ensure_terminal_punctuation("，".join(chunks))


def clean_flickr_manual_text(text: str) -> str:
    """还原 zhm 文本：删除 BosonNLP 的 :POS 后缀并连接中文 token。"""
    text = remove_control_characters(unicodedata.normalize("NFKC", text)).strip()
    words = [POS_SUFFIX_PATTERN.sub("", token) for token in text.split()]
    return ensure_terminal_punctuation("".join(word for word in words if word))


def coco_image_id(source_id: str) -> int:
    match = COCO_ID_PATTERN.fullmatch(source_id)
    if not match:
        raise DataError(f"非法 COCO 图片 ID：{source_id!r}")
    prefix = "1" if match.group(1) == "train" else "2"
    return int(prefix + match.group(2))


def warn_expected(actual: int, expected: int, label: str, issues: IssueTracker) -> None:
    if actual != expected:
        issues.add("baseline_count_mismatch", f"{label}：期望 {expected}，实际 {actual}")


def parse_numeric_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in strict_lines(path) if line.strip()]
    if any(not image_id.isdigit() for image_id in ids):
        raise DataError(f"图片 ID 必须全部是十进制整数：{path}")
    if len(ids) != len(set(ids)):
        raise DataError(f"图片划分文件存在重复 ID：{path}")
    return ids


def deduplicate_training(entries: list[TextEntry]) -> tuple[list[TextEntry], int]:
    result = []
    seen = set()
    removed = 0
    for entry in entries:
        key = (entry.image_id, entry.text)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        result.append(entry)
    return result, removed


def load_flickr(raw_root: Path) -> DatasetData:
    base = raw_root / "Flickr30k-CN" / "extracted"
    annotations = base / "annotations"
    image_root = base / "images" / "flickr30k-images"
    folders = {
        "train": "flickr30kzhbbosontrain",
        "valid": "flickr30kzhbbosonval",
        "test": "flickr30kzhmbosontest",
    }
    issues = IssueTracker()
    splits = {split: SplitData() for split in SPLITS}
    split_source_ids: dict[str, list[str]] = {}

    for split, folder in folders.items():
        split_root = annotations / folder
        ids_path = split_root / "ImageSets" / f"{folder}.txt"
        caption_path = split_root / "TextData" / f"seg.{folder}.caption.txt"
        for required_path in (ids_path, caption_path, image_root):
            if not required_path.exists():
                raise DataError(f"缺少 Flickr 源文件或目录：{required_path}")

        source_ids = parse_numeric_ids(ids_path)
        warn_expected(
            len(source_ids), EXPECTED["Flickr30k-CN"]["images"][split], f"Flickr {split} 图片", issues
        )
        split_source_ids[split] = source_ids
        splits[split].images = [
            ImageEntry(source_id=image_id, image_id=int(image_id), path=image_root / f"{image_id}.jpg")
            for image_id in source_ids
        ]

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = set(split_source_ids[left]) & set(split_source_ids[right])
        if overlap:
            raise DataError(f"Flickr {left}/{right} 图片 ID 跨划分重叠：{sorted(overlap)[:5]}")

    expected_image_files = set().union(*(set(values) for values in split_source_ids.values()))
    actual_image_files = {path.stem for path in image_root.glob("*.jpg")}
    for image_id in sorted(expected_image_files - actual_image_files):
        issues.add("missing_image_file", f"Flickr 划分引用但图片不存在：{image_id}.jpg")
    for image_id in sorted(actual_image_files - expected_image_files):
        issues.add("orphan_image_file", f"Flickr 图片不属于任何划分，忽略：{image_id}.jpg")

    next_text_id = 0
    for split, folder in folders.items():
        caption_path = annotations / folder / "TextData" / f"seg.{folder}.caption.txt"
        allowed_ids = set(split_source_ids[split])
        expected_language = "zhm" if split == "test" else "zhb"
        seen_source_text_ids = set()
        parsed_entries = []
        for line_number, line in enumerate(strict_lines(caption_path), start=1):
            try:
                source_text_id, raw_text = line.split(" ", 1)
                source_image_id, language, caption_number = source_text_id.split("#")
            except ValueError:
                issues.add("malformed_caption", f"{caption_path}:{line_number}: {line!r}")
                next_text_id += 1
                continue
            if source_image_id not in allowed_ids:
                issues.add("caption_without_split_mapping", f"Flickr 标注不属于当前划分：{source_text_id}")
                next_text_id += 1
                continue
            if language != expected_language or not caption_number.isdigit():
                issues.add("malformed_caption_id", f"Flickr caption ID 格式异常：{source_text_id}")
                next_text_id += 1
                continue
            if source_text_id in seen_source_text_ids:
                issues.add("duplicate_caption_id", f"Flickr caption ID 重复，保留首次：{source_text_id}")
                next_text_id += 1
                continue
            seen_source_text_ids.add(source_text_id)
            cleaner = clean_flickr_manual_text if split == "test" else clean_flickr_machine_text
            clean_text = cleaner(raw_text)
            if not clean_text:
                issues.add("empty_clean_text", f"Flickr 清洗后文本为空：{source_text_id}")
                next_text_id += 1
                continue
            parsed_entries.append(
                TextEntry(
                    source_text_id=source_text_id,
                    text_id=next_text_id,
                    image_id=int(source_image_id),
                    raw_text=raw_text,
                    text=clean_text,
                )
            )
            next_text_id += 1

        splits[split].source_text_count = len(parsed_entries)
        warn_expected(
            len(parsed_entries),
            EXPECTED["Flickr30k-CN"]["texts_before_dedup"][split],
            f"Flickr {split} 原始文本",
            issues,
        )
        described_ids = {str(entry.image_id) for entry in parsed_entries}
        for image_id in sorted(allowed_ids - described_ids):
            issues.add("image_without_caption", f"Flickr {split} 图片没有可用描述：{image_id}")
        if split == "train":
            parsed_entries, removed = deduplicate_training(parsed_entries)
            splits[split].removed_duplicate_count = removed
        splits[split].texts = parsed_entries
        warn_expected(
            len(parsed_entries),
            EXPECTED["Flickr30k-CN"]["texts_after_dedup"][split],
            f"Flickr {split} 清洗后文本",
            issues,
        )

    return DatasetData(name="Flickr30k-CN", splits=splits, issues=issues)


def load_coco(raw_root: Path) -> DatasetData:
    base = raw_root / "COCO-CN" / "extracted"
    annotation_root = base / "coco-cn-version1805v1.1"
    image_root = base / "images"
    split_files = {
        "train": "coco-cn_train.txt",
        "valid": "coco-cn_val.txt",
        "test": "coco-cn_test.txt",
    }
    caption_path = annotation_root / "imageid.human-written-caption.txt"
    issues = IssueTracker()
    for required_path in (annotation_root, image_root / "train2014", image_root / "val2014", caption_path):
        if not required_path.exists():
            raise DataError(f"缺少 COCO-CN 源文件或目录：{required_path}")

    split_source_ids: dict[str, list[str]] = {}
    for split, filename in split_files.items():
        path = annotation_root / filename
        source_ids = []
        seen_source_ids = set()
        for line_number, line in enumerate(strict_lines(path), start=1):
            source_id = line.strip()
            if not source_id:
                continue
            try:
                coco_image_id(source_id)
            except DataError:
                issues.add("invalid_image_id", f"{path}:{line_number}: {source_id!r}")
                continue
            if source_id in seen_source_ids:
                issues.add("duplicate_split_image_id", f"{path}:{line_number}: {source_id}")
                continue
            seen_source_ids.add(source_id)
            source_ids.append(source_id)
        warn_expected(len(source_ids), EXPECTED["COCO-CN"]["images"][split], f"COCO {split} 图片", issues)
        split_source_ids[split] = source_ids

    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = set(split_source_ids[left]) & set(split_source_ids[right])
        if overlap:
            raise DataError(f"COCO {left}/{right} 图片 ID 跨划分重叠：{sorted(overlap)[:5]}")

    source_id_to_split = {
        source_id: split for split, source_ids in split_source_ids.items() for source_id in source_ids
    }
    captions_by_image: dict[str, list[tuple[str, str]]] = defaultdict(list)
    seen_caption_ids = set()
    for line_number, line in enumerate(strict_lines(caption_path), start=1):
        try:
            source_text_id, raw_text = line.split("\t", 1)
            source_image_id, caption_number = source_text_id.rsplit("#", 1)
        except ValueError:
            issues.add("malformed_caption", f"{caption_path}:{line_number}: {line!r}")
            continue
        if source_text_id in seen_caption_ids:
            issues.add("duplicate_caption_id", f"COCO-CN caption ID 重复，保留首次：{source_text_id}")
            continue
        if source_image_id not in source_id_to_split or not caption_number.isdigit():
            issues.add("caption_without_split_mapping", f"COCO-CN 人工描述无法映射：{source_text_id}")
            continue
        seen_caption_ids.add(source_text_id)
        captions_by_image[source_image_id].append((source_text_id, raw_text))

    warn_expected(len(seen_caption_ids), 22_218, "COCO-CN 人工描述总数", issues)

    splits = {split: SplitData() for split in SPLITS}
    next_text_id = 1
    expected_image_paths = set()
    for split in SPLITS:
        parsed_entries = []
        for source_image_id in split_source_ids[split]:
            match = COCO_ID_PATTERN.fullmatch(source_image_id)
            assert match is not None
            source_partition = "train2014" if match.group(1) == "train" else "val2014"
            image_path = image_root / source_partition / f"{source_image_id}.jpg"
            expected_image_paths.add(image_path)
            image_id = coco_image_id(source_image_id)
            splits[split].images.append(ImageEntry(source_id=source_image_id, image_id=image_id, path=image_path))
            image_captions = captions_by_image.get(source_image_id, [])
            if not image_captions:
                issues.add("image_without_caption", f"COCO-CN 图片没有人工描述：{source_image_id}")
            for source_text_id, raw_text in image_captions:
                clean_text = normalize_common(raw_text)
                if not clean_text:
                    issues.add("empty_clean_text", f"COCO-CN 清洗后文本为空：{source_text_id}")
                    next_text_id += 1
                    continue
                parsed_entries.append(
                    TextEntry(
                        source_text_id=source_text_id,
                        text_id=next_text_id,
                        image_id=image_id,
                        raw_text=raw_text,
                        text=clean_text,
                    )
                )
                next_text_id += 1

        splits[split].source_text_count = len(parsed_entries)
        warn_expected(
            len(parsed_entries),
            EXPECTED["COCO-CN"]["texts_before_dedup"][split],
            f"COCO {split} 原始文本",
            issues,
        )
        if split == "train":
            parsed_entries, removed = deduplicate_training(parsed_entries)
            splits[split].removed_duplicate_count = removed
        splits[split].texts = parsed_entries
        warn_expected(
            len(parsed_entries),
            EXPECTED["COCO-CN"]["texts_after_dedup"][split],
            f"COCO {split} 清洗后文本",
            issues,
        )

    actual_image_paths = set(image_root.glob("*/*.jpg"))
    for path in sorted(expected_image_paths - actual_image_paths):
        issues.add("missing_image_file", f"COCO-CN 划分引用但图片不存在：{path}")
    for path in sorted(actual_image_paths - expected_image_paths):
        issues.add("orphan_image_file", f"COCO-CN 图片不属于正式划分，忽略：{path}")
    return DatasetData(name="COCO-CN", splits=splits, issues=issues)


def inspect_image(entry: ImageEntry) -> tuple[str | None, str | None, int | None, int | None, str | None, str | None]:
    if not entry.path.is_file() or entry.path.stat().st_size == 0:
        return None, None, None, None, None, f"图片不存在或为空：{entry.path}"
    try:
        with Image.open(entry.path) as image:
            image_format = image.format or "UNKNOWN"
            mode = image.mode
            width, height = image.size
            image.verify()
        with Image.open(entry.path) as image:
            image.load()
    except Exception as exc:
        return None, None, None, None, None, f"图片无法完整解码：{entry.path}: {exc}"
    if min(width, height) < 32:
        return None, None, None, None, None, f"图片短边小于 32 像素：{entry.path} ({width}x{height})"
    digest = hashlib.sha256(entry.path.read_bytes()).hexdigest()
    return image_format, mode, width, height, digest, None


def percentile(values: list[int], ratio: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]


def validate_dataset(dataset: DatasetData, workers: int, sample_count: int) -> None:
    all_image_ids = set()
    all_text_ids = set()
    image_locations: dict[int, str] = {}
    image_entries = []
    for split in SPLITS:
        split_data = dataset.splits[split]
        split_image_ids = {entry.image_id for entry in split_data.images}
        if len(split_image_ids) != len(split_data.images):
            raise DataError(f"{dataset.name} {split} 输出图片 ID 重复")
        overlap = all_image_ids & split_image_ids
        if overlap:
            raise DataError(f"{dataset.name} 图片 ID 跨划分冲突：{sorted(overlap)[:5]}")
        all_image_ids.update(split_image_ids)
        for entry in split_data.images:
            image_locations[entry.image_id] = split
            image_entries.append(entry)

        described = set()
        for entry in split_data.texts:
            if entry.text_id in all_text_ids:
                raise DataError(f"{dataset.name} text_id 重复：{entry.text_id}")
            all_text_ids.add(entry.text_id)
            if entry.image_id not in split_image_ids:
                raise DataError(f"{dataset.name} {split} 文本引用不存在的图片：{entry.source_text_id}")
            if not entry.text or any(unicodedata.category(c).startswith("C") for c in entry.text):
                raise DataError(f"{dataset.name} 文本为空或仍含控制字符：{entry.source_text_id}")
            described.add(entry.image_id)
        if described != split_image_ids:
            missing_descriptions = split_image_ids - described
            for image_id in sorted(missing_descriptions):
                dataset.issues.add("image_without_caption", f"{dataset.name} {split} 图片无可用文本，跳过：{image_id}")
            split_data.images = [entry for entry in split_data.images if entry.image_id in described]
            split_image_ids = described

    image_entries = [entry for split in SPLITS for entry in dataset.splits[split].images]
    image_locations = {
        entry.image_id: split for split in SPLITS for entry in dataset.splits[split].images
    }
    print(f"[{dataset.name}] 正在完整解码并计算 {len(image_entries)} 张图片的 SHA256...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inspection_results = list(pool.map(inspect_image, image_entries))

    hash_to_entries: dict[str, list[ImageEntry]] = defaultdict(list)
    formats = Counter()
    modes = Counter()
    widths = []
    heights = []
    invalid_image_ids = set()
    for entry, (image_format, mode, width, height, digest, error) in zip(image_entries, inspection_results):
        if error:
            invalid_image_ids.add(entry.image_id)
            dataset.issues.add("invalid_image", error)
            continue
        assert image_format is not None and mode is not None
        assert width is not None and height is not None and digest is not None
        entry.sha256 = digest
        hash_to_entries[digest].append(entry)
        formats[image_format] += 1
        modes[mode] += 1
        widths.append(width)
        heights.append(height)

    if invalid_image_ids:
        for split in SPLITS:
            split_data = dataset.splits[split]
            split_data.images = [entry for entry in split_data.images if entry.image_id not in invalid_image_ids]
            split_data.texts = [entry for entry in split_data.texts if entry.image_id not in invalid_image_ids]
        image_entries = [entry for entry in image_entries if entry.image_id not in invalid_image_ids]

    for split in SPLITS:
        if not dataset.splits[split].images or not dataset.splits[split].texts:
            raise DataError(f"{dataset.name} {split} 在跳过异常记录后已无可用图文数据")

    same_bytes_groups = [entries for entries in hash_to_entries.values() if len(entries) > 1]
    for entries in same_bytes_groups:
        locations = {image_locations[entry.image_id] for entry in entries}
        if len(locations) > 1:
            raise DataError(
                f"{dataset.name} 发现跨划分完全相同图片："
                + ", ".join(f"{image_locations[e.image_id]}:{e.source_id}" for e in entries)
            )

    print(
        f"[{dataset.name}] 图片通过：格式={dict(formats)}，模式={dict(modes)}，"
        f"宽={min(widths)}..{max(widths)}，高={min(heights)}..{max(heights)}，"
        f"同划分重复字节组={len(same_bytes_groups)}"
    )

    randomizer = random.Random(20260731)
    for split in SPLITS:
        split_data = dataset.splits[split]
        lengths = [len(entry.text) for entry in split_data.texts]
        print(
            f"[{dataset.name}/{split}] images={len(split_data.images)}，"
            f"texts={len(split_data.texts)}，source_texts={split_data.source_text_count}，"
            f"train_duplicates_removed={split_data.removed_duplicate_count}，"
            f"text_length(min/p50/p95/p99/max)="
            f"{min(lengths)}/{int(statistics.median(lengths))}/{percentile(lengths, .95)}/"
            f"{percentile(lengths, .99)}/{max(lengths)}"
        )
        if sample_count:
            print(f"[{dataset.name}/{split}] 固定种子抽样：")
            for entry in randomizer.sample(split_data.texts, min(sample_count, len(split_data.texts))):
                image_path = next(image.path for image in split_data.images if image.image_id == entry.image_id)
                print(f"  image={image_path}")
                print(f"  source={entry.raw_text}")
                print(f"  clean ={entry.text}")
    dataset.issues.print_summary(dataset.name)


def write_dataset_files(dataset: DatasetData, temporary_directory: Path) -> None:
    temporary_directory.mkdir(parents=True, exist_ok=False)
    for split in SPLITS:
        split_data = dataset.splits[split]
        image_output = temporary_directory / f"{split}_imgs.tsv"
        text_output = temporary_directory / f"{split}_texts.jsonl"
        with image_output.open("w", encoding="utf-8", newline="\n") as output:
            for image in split_data.images:
                encoded = base64.b64encode(image.path.read_bytes()).decode("ascii")
                output.write(f"{image.image_id}\t{encoded}\n")
        with text_output.open("w", encoding="utf-8", newline="\n") as output:
            for text in split_data.texts:
                obj = {"text_id": text.text_id, "text": text.text, "image_ids": [text.image_id]}
                output.write(json.dumps(obj, ensure_ascii=False) + "\n")


def validate_written_files(dataset: DatasetData, temporary_directory: Path) -> None:
    global_text_ids = set()
    for split in SPLITS:
        split_data = dataset.splits[split]
        source_by_id = {entry.image_id: entry for entry in split_data.images}
        seen_images = set()
        image_path = temporary_directory / f"{split}_imgs.tsv"
        with image_path.open("r", encoding="utf-8", errors="strict") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 2:
                    raise DataError(f"输出 TSV 不是两列：{image_path}:{line_number}")
                image_id_text, encoded = fields
                if not image_id_text.isdigit():
                    raise DataError(f"输出 TSV 图片 ID 不是整数：{image_path}:{line_number}")
                image_id = int(image_id_text)
                if image_id in seen_images or image_id not in source_by_id:
                    raise DataError(f"输出 TSV 图片 ID 重复或未知：{image_id}")
                seen_images.add(image_id)
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except Exception as exc:
                    raise DataError(f"输出 TSV Base64 非法：{image_path}:{line_number}: {exc}") from exc
                expected_hash = source_by_id[image_id].sha256
                if hashlib.sha256(decoded).hexdigest() != expected_hash:
                    raise DataError(f"输出 TSV 图片字节与源文件不同：{image_id}")
        if seen_images != set(source_by_id):
            raise DataError(f"输出 TSV 图片集合不完整：{image_path}")

        seen_texts = 0
        text_path = temporary_directory / f"{split}_texts.jsonl"
        with text_path.open("r", encoding="utf-8", errors="strict") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"输出 JSONL 非法：{text_path}:{line_number}: {exc}") from exc
                if set(obj) != {"text_id", "text", "image_ids"}:
                    raise DataError(f"输出 JSONL 字段不正确：{text_path}:{line_number}")
                if not isinstance(obj["text_id"], int) or obj["text_id"] in global_text_ids:
                    raise DataError(f"输出 text_id 类型错误或重复：{obj['text_id']!r}")
                if not isinstance(obj["text"], str) or not obj["text"]:
                    raise DataError(f"输出 text 字段错误：{text_path}:{line_number}")
                if (
                    not isinstance(obj["image_ids"], list)
                    or len(obj["image_ids"]) != 1
                    or not isinstance(obj["image_ids"][0], int)
                    or obj["image_ids"][0] not in seen_images
                ):
                    raise DataError(f"输出 image_ids 字段错误：{text_path}:{line_number}")
                global_text_ids.add(obj["text_id"])
                seen_texts += 1
        if seen_texts != len(split_data.texts):
            raise DataError(f"输出 JSONL 行数异常：{text_path}: {seen_texts} != {len(split_data.texts)}")

    produced_files = sorted(path.name for path in temporary_directory.iterdir() if path.is_file())
    expected_files = sorted(f"{split}_{kind}.{extension}" for split in SPLITS for kind, extension in (("imgs", "tsv"), ("texts", "jsonl")))
    if produced_files != expected_files:
        raise DataError(f"输出文件集合异常：{produced_files}")


def build_dataset(dataset: DatasetData, output_root: Path, overwrite: bool) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / dataset.name
    temporary = output_root / f".{dataset.name}.tmp-{os.getpid()}"
    backup = output_root / f".{dataset.name}.backup-{os.getpid()}"
    if target.exists() and not overwrite:
        raise DataError(f"输出目录已存在；如确认替换请增加 --overwrite：{target}")
    if temporary.exists() or backup.exists():
        raise DataError(f"发现本次运行的临时目录冲突：{temporary} 或 {backup}")

    try:
        print(f"[{dataset.name}] 写入临时输出：{temporary}")
        write_dataset_files(dataset, temporary)
        print(f"[{dataset.name}] 对六个输出文件的字段、Base64 和源字节执行全量回读验证...")
        validate_written_files(dataset, temporary)
        if target.exists():
            target.replace(backup)
        temporary.replace(target)
        if backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup.exists() and not target.exists():
            backup.replace(target)
        raise
    print(f"[{dataset.name}] 构建完成：{target}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "build"), help="仅检查，或检查通过后构建 TSV/JSONL")
    parser.add_argument(
        "--dataset",
        choices=("all", "flickr30k-cn", "coco-cn"),
        default="all",
        help="处理的数据集，默认 all",
    )
    parser.add_argument("--raw-root", type=Path, default=workspace_root / "datasets_raw")
    parser.add_argument("--output-root", type=Path, default=workspace_root / "datasets_rebuilt")
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--sample-count", type=int, default=3, help="每个划分打印的固定种子抽样数")
    parser.add_argument("--overwrite", action="store_true", help="原子替换已存在的对应重建目录")
    args = parser.parse_args()
    if args.workers < 1 or args.sample_count < 0:
        parser.error("--workers 必须 >= 1，--sample-count 必须 >= 0")
    return args


def main() -> int:
    args = parse_args()
    loaders: list[Callable[[Path], DatasetData]] = []
    if args.dataset in ("all", "flickr30k-cn"):
        loaders.append(load_flickr)
    if args.dataset in ("all", "coco-cn"):
        loaders.append(load_coco)

    try:
        for loader in loaders:
            dataset = loader(args.raw_root)
            validate_dataset(dataset, workers=args.workers, sample_count=args.sample_count)
            if args.action == "build":
                build_dataset(dataset, args.output_root, overwrite=args.overwrite)
    except DataError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
