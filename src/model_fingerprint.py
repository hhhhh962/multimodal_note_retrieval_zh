"""模型、数据清单、向量状态与索引之间的一致性保护。

模型替换后继续写旧向量通常不会立即报错，却会生成语义空间混杂的损坏索引。本模块把
模型内容指纹、数据清单哈希、维度和归一化方式定义成统一契约；续跑或加载索引前必须
显式校验，任一字段不同都会抛出 ``MetadataMismatchError``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FINGERPRINT_FILENAME = "model_fingerprint.json"
REQUIRED_COMPATIBILITY_FIELDS = (
    "model_fingerprint",
    "data_manifest_hash",
    "dimension",
    "normalization",
)
EXCLUDED_MODEL_FILES = {
    FINGERPRINT_FILENAME,
    "export_manifest.json",
    "embedding_meta.json",
    "index_meta.json",
}
MODEL_ARTIFACT_NAMES = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "retrieval_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "sentencepiece.bpe.model",
    "spiece.model",
}
MODEL_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


class MetadataMismatchError(RuntimeError):
    """表示旧产物与本次模型或数据不兼容，禁止断点续用。"""


def sha256_file(path: str | Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """分块计算文件 SHA-256，避免模型权重整体读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_model_artifact(path: Path) -> bool:
    """仅选择决定 HF 模型/processor 行为的文件，排除日志与循环元数据。"""

    if path.name in EXCLUDED_MODEL_FILES:
        return False
    if path.name in MODEL_ARTIFACT_NAMES:
        return True
    if path.name.endswith(MODEL_WEIGHT_SUFFIXES):
        return True
    if path.name in {"model.safetensors.index.json", "pytorch_model.bin.index.json"}:
        return True
    if path.name.startswith(("model-", "pytorch_model-")):
        return path.suffix in {".json", ".safetensors", ".bin"}
    return False


def fingerprint_model_directory(model_dir: str | Path) -> dict[str, Any]:
    """根据完整模型与 processor 工件内容构造可迁移的确定性指纹。"""

    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and _is_model_artifact(path)),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"模型目录中没有可识别的 HF 模型工件：{root}")
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path)
        size = int(path.stat().st_size)
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(b"\n")
        entries.append({"path": relative, "size": size, "sha256": file_digest})
    return {
        "algorithm": "sha256",
        "value": aggregate.hexdigest(),
        "files": entries,
    }


def compute_model_fingerprint(model_dir: str | Path) -> str:
    """返回模型目录指纹字符串，便于写入向量和索引元数据。"""

    return str(fingerprint_model_directory(model_dir)["value"])


def write_model_fingerprint(model_dir: str | Path) -> dict[str, Any]:
    """计算并原子式写入模型目录的 ``model_fingerprint.json``。"""

    root = Path(model_dir)
    payload = fingerprint_model_directory(root)
    target = root / FINGERPRINT_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return payload


def read_model_fingerprint(model_dir: str | Path, verify: bool = True) -> dict[str, Any]:
    """读取已保存指纹，并可重新计算内容以发现权重被替换。"""

    root = Path(model_dir)
    path = root / FINGERPRINT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"缺少模型指纹文件：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        current = fingerprint_model_directory(root)
        if payload.get("value") != current.get("value"):
            raise MetadataMismatchError(
                f"模型目录内容与已保存指纹不一致：saved={payload.get('value')} current={current.get('value')}"
            )
    return payload


def _canonical_json_hash(value: Any) -> str:
    """对 JSON 语义值执行稳定序列化后求 SHA-256。"""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe_manifest_value(value: Any) -> Any:
    """把清单中的路径和无序集合转换成稳定 JSON 值。"""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_manifest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe_manifest_value(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_json_safe_manifest_value(item) for item in value]
    return value


def compute_data_manifest_hash(manifest: Any) -> str:
    """计算数据清单哈希。

    单文件按原始字节计算；多个文件建议传 ``{逻辑名: 路径}``，聚合值只包含逻辑名、
    大小与内容哈希，因此数据目录迁移不会改变结果。
    """

    if isinstance(manifest, Path) or isinstance(manifest, str):
        path = Path(manifest)
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if is_file:
            return sha256_file(path)
        if isinstance(manifest, Path):
            raise FileNotFoundError(f"数据清单不存在：{path}")
        return _canonical_json_hash(_json_safe_manifest_value(manifest))
    if isinstance(manifest, Mapping):
        canonical: dict[str, Any] = {}
        for key in sorted(manifest, key=lambda item: str(item)):
            value = manifest[key]
            try:
                value_is_file = isinstance(value, (str, Path)) and Path(value).is_file()
            except OSError:
                value_is_file = False
            if value_is_file:
                path = Path(value)
                canonical[str(key)] = {
                    "size": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            else:
                canonical[str(key)] = _json_safe_manifest_value(value)
        return _canonical_json_hash(canonical)
    if isinstance(manifest, Iterable) and not isinstance(manifest, (bytes, bytearray)):
        values = list(manifest)
        canonical_values: list[Any] = []
        for value in values:
            try:
                value_is_file = isinstance(value, (str, Path)) and Path(value).is_file()
            except OSError:
                value_is_file = False
            if value_is_file:
                path = Path(value)
                canonical_values.append({"size": int(path.stat().st_size), "sha256": sha256_file(path)})
            else:
                canonical_values.append(_json_safe_manifest_value(value))
        return _canonical_json_hash(canonical_values)
    return _canonical_json_hash(_json_safe_manifest_value(manifest))


def build_compatibility_metadata(
    model_fingerprint: str,
    data_manifest_hash: str,
    dimension: int,
    normalization: str = "l2",
    **extra: Any,
) -> dict[str, Any]:
    """构造向量状态与索引共同使用的兼容性元数据。"""

    if int(dimension) <= 0:
        raise ValueError("dimension 必须是正整数")
    if not model_fingerprint:
        raise ValueError("model_fingerprint 不能为空")
    if not data_manifest_hash:
        raise ValueError("data_manifest_hash 不能为空")
    payload: dict[str, Any] = {
        "model_fingerprint": str(model_fingerprint),
        "data_manifest_hash": str(data_manifest_hash),
        "dimension": int(dimension),
        "normalization": str(normalization).lower(),
    }
    payload.update(extra)
    return payload


def compatibility_mismatches(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """返回四个强制字段的差异，缺失字段也视为不兼容。"""

    mismatches: dict[str, dict[str, Any]] = {}
    for field in REQUIRED_COMPATIBILITY_FIELDS:
        if field not in existing or field not in expected:
            mismatches[field] = {
                "existing": existing.get(field, "<missing>"),
                "expected": expected.get(field, "<missing>"),
            }
            continue
        old_value = existing.get(field)
        new_value = expected.get(field)
        if field == "dimension":
            try:
                old_value = int(old_value)
            except (TypeError, ValueError):
                pass
            try:
                new_value = int(new_value)
            except (TypeError, ValueError):
                pass
        if field == "normalization":
            old_value = str(old_value).lower() if old_value is not None else None
            new_value = str(new_value).lower() if new_value is not None else None
        if old_value != new_value:
            mismatches[field] = {"existing": old_value, "expected": new_value}
    return mismatches


def assert_metadata_compatible(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
    context: str = "断点续跑",
) -> None:
    """若模型、数据、维度或归一化任一不一致，立即阻止旧产物复用。"""

    mismatches = compatibility_mismatches(existing, expected)
    if mismatches:
        detail = "; ".join(
            f"{field}: existing={values['existing']!r}, expected={values['expected']!r}"
            for field, values in mismatches.items()
        )
        raise MetadataMismatchError(f"{context}元数据不兼容，禁止复用旧产物：{detail}")


def load_and_assert_metadata(
    path: str | Path,
    expected: Mapping[str, Any],
    context: str = "断点续跑",
) -> dict[str, Any]:
    """从 JSON 文件读取元数据并执行强制兼容性检查。"""

    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"缺少{context}元数据：{metadata_path}")
    existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(existing, Mapping):
        raise ValueError(f"元数据根节点必须是对象：{metadata_path}")
    assert_metadata_compatible(existing, expected, context=context)
    return dict(existing)


def write_compatibility_metadata(path: str | Path, metadata: Mapping[str, Any]) -> Path:
    """校验强制字段齐全后写出向量或索引元数据。"""

    missing = [field for field in REQUIRED_COMPATIBILITY_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"兼容性元数据缺少字段：{', '.join(missing)}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
