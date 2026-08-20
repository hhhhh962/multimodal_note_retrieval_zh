"""Chinese-CLIP 微调使用的 LMDB 数据读取与多正样本组批工具。

Chinese-CLIP 官方 LMDB 格式在 ``pairs`` 中保存序列化后的
``(image_id, text_id, raw_text)`` 元组，在 ``imgs`` 中保存 Base64 编码的图片
字节。数据集构造时不会持有 LMDB 句柄；每个 DataLoader worker 首次读取样本时，
才分别打开自己的只读环境和事务。
"""

from __future__ import annotations

import base64
import os
import pickle
import random
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

try:
    import lmdb
except ImportError:  # pragma: no cover - 仅在尚未安装训练依赖时发生。
    lmdb = None


DEFAULT_EPOCH_SIZE = 419_294
DEFAULT_SEED = 20_260_731


def make_uid(source: str, raw_id: Any) -> str:
    """为原始 ID 增加来源前缀，避免不同数据集之间发生 ID 碰撞。"""

    source = str(source).strip()
    if not source:
        raise ValueError("source must be a non-empty string")
    if isinstance(raw_id, bytes):
        raw_id = raw_id.decode("utf-8")
    return f"{source}:{raw_id}"


def build_positive_relations(
    pairs: Iterable[tuple[str, str]],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """根据 ``(text_uid, image_uid)`` 构建完整的双向正样本关系图。"""

    text_to_images_mutable: dict[str, set[str]] = defaultdict(set)
    image_to_texts_mutable: dict[str, set[str]] = defaultdict(set)
    for text_uid, image_uid in pairs:
        text_uid = str(text_uid)
        image_uid = str(image_uid)
        text_to_images_mutable[text_uid].add(image_uid)
        image_to_texts_mutable[image_uid].add(text_uid)

    text_to_images = {
        key: frozenset(values) for key, values in text_to_images_mutable.items()
    }
    image_to_texts = {
        key: frozenset(values) for key, values in image_to_texts_mutable.items()
    }
    return text_to_images, image_to_texts


def merge_positive_relations(
    datasets: Iterable[Any],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """合并多个来源数据集公开的完整正样本关系映射。"""

    pairs: list[tuple[str, str]] = []
    for dataset in datasets:
        for text_uid, image_uids in dataset.text_to_images.items():
            pairs.extend((text_uid, image_uid) for image_uid in image_uids)
    return build_positive_relations(pairs)


def _require_lmdb() -> Any:
    if lmdb is None:
        raise ImportError(
            "LMDB support requires lmdb==1.3.0; install the training dependencies first."
        )
    return lmdb


def _as_bytes(value: Any, *, key: str) -> bytes:
    if value is None:
        raise KeyError(f"LMDB key {key!r} was not found")
    return value.tobytes() if hasattr(value, "tobytes") else bytes(value)


class LMDBPairDataset(Dataset):
    """读取一个 Chinese-CLIP 原始数据划分，并保留全部图文正样本关系。

    参数：
        lmdb_path: 同时包含 ``pairs`` 与 ``imgs`` LMDB 的划分目录，通常为
            ``<dataset>/lmdb/train``。
        source: 稳定的数据来源名称，例如 ``MUGE``、``Flickr30k-CN`` 或
            ``COCO-CN``；该名称会添加到所有原始 ID 之前。
        image_transform: 可选的图片变换函数，输入为解码后的 RGB PIL 图片。
            训练入口可注入轻度随机裁剪和水平翻转，评估入口可注入固定的双三次
            缩放和中心裁剪。
        build_relations: 是否在构造时扫描一次图文对元数据，生成完整的文搜图与
            图搜文正样本关系映射。
    """

    def __init__(
        self,
        lmdb_path: str | os.PathLike[str],
        source: str,
        image_transform: Callable[[Image.Image], Any] | None = None,
        *,
        build_relations: bool = True,
    ) -> None:
        super().__init__()
        _require_lmdb()

        self.lmdb_path = Path(lmdb_path)
        self.pairs_path = self.lmdb_path / "pairs"
        self.images_path = self.lmdb_path / "imgs"
        self.source = str(source).strip()
        if not self.source:
            raise ValueError("source must be a non-empty string")
        if not self.pairs_path.is_dir():
            raise FileNotFoundError(f"pair LMDB does not exist: {self.pairs_path}")
        if not self.images_path.is_dir():
            raise FileNotFoundError(f"image LMDB does not exist: {self.images_path}")

        self.image_transform = image_transform
        self.number_samples = self._read_count(self.pairs_path, b"num_samples")
        self.number_images = self._read_count(self.images_path, b"num_images")

        # 构造时不打开运行期句柄；直到 DataLoader worker 首次执行 __getitem__
        # 才打开。num_workers == 0 时则由调用进程首次读取时打开。
        self._pairs_env: Any | None = None
        self._pairs_txn: Any | None = None
        self._images_env: Any | None = None
        self._images_txn: Any | None = None
        self._open_pid: int | None = None

        if build_relations:
            self.text_to_images, self.image_to_texts = self._scan_relations()
        else:
            self.text_to_images = {}
            self.image_to_texts = {}
        # 同时提供单数形式别名，并通过同一对象保证两者始终同步。
        self.image_to_text = self.image_to_texts

    @staticmethod
    def _read_count(path: Path, key: bytes) -> int:
        lmdb_module = _require_lmdb()
        env = lmdb_module.open(
            str(path),
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        try:
            with env.begin(buffers=True) as txn:
                value = _as_bytes(txn.get(key), key=key.decode("utf-8"))
            count = int(value.decode("utf-8"))
        finally:
            env.close()
        if count < 0:
            raise ValueError(f"negative item count in {path}: {count}")
        return count

    @staticmethod
    def _decode_pair(value: Any, *, index: int) -> tuple[Any, Any, str]:
        payload = _as_bytes(value, key=str(index))
        pair = pickle.loads(payload)
        if not isinstance(pair, (tuple, list)) or len(pair) != 3:
            raise ValueError(
                f"pair LMDB entry {index} must contain (image_id, text_id, text)"
            )
        image_id, text_id, raw_text = pair
        if isinstance(raw_text, bytes):
            raw_text = raw_text.decode("utf-8")
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        return image_id, text_id, raw_text

    def _scan_relations(
        self,
    ) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
        lmdb_module = _require_lmdb()
        env = lmdb_module.open(
            str(self.pairs_path),
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
        )
        relations: list[tuple[str, str]] = []
        try:
            with env.begin(buffers=True) as txn:
                for index in range(self.number_samples):
                    image_id, text_id, _ = self._decode_pair(
                        txn.get(str(index).encode("utf-8")), index=index
                    )
                    relations.append(
                        (make_uid(self.source, text_id), make_uid(self.source, image_id))
                    )
        finally:
            env.close()
        return build_positive_relations(relations)

    def _ensure_open(self) -> None:
        current_pid = os.getpid()
        if self._open_pid == current_pid and self._pairs_txn is not None:
            return
        self.close()
        lmdb_module = _require_lmdb()
        open_options = dict(
            readonly=True,
            create=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=512,
        )
        self._pairs_env = lmdb_module.open(str(self.pairs_path), **open_options)
        self._images_env = lmdb_module.open(str(self.images_path), **open_options)
        self._pairs_txn = self._pairs_env.begin(buffers=True)
        self._images_txn = self._images_env.begin(buffers=True)
        self._open_pid = current_pid

    def close(self) -> None:
        """关闭当前进程已经打开的只读事务与 LMDB 环境。"""

        for transaction_name in ("_pairs_txn", "_images_txn"):
            transaction = getattr(self, transaction_name, None)
            if transaction is not None:
                try:
                    transaction.abort()
                except Exception:
                    pass
                setattr(self, transaction_name, None)
        for environment_name in ("_pairs_env", "_images_env"):
            environment = getattr(self, environment_name, None)
            if environment is not None:
                try:
                    environment.close()
                except Exception:
                    pass
                setattr(self, environment_name, None)
        self._open_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for name in (
            "_pairs_env",
            "_pairs_txn",
            "_images_env",
            "_images_txn",
            "_open_pid",
        ):
            state[name] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - 解释器退出时的析构顺序不确定。
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.number_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += self.number_samples
        if index < 0 or index >= self.number_samples:
            raise IndexError(index)
        self._ensure_open()

        image_id, text_id, raw_text = self._decode_pair(
            self._pairs_txn.get(str(index).encode("utf-8")), index=index
        )
        image_key = str(image_id).encode("utf-8")
        encoded_image = _as_bytes(
            self._images_txn.get(image_key), key=str(image_id)
        ).strip()
        # urlsafe_b64decode 同时兼容普通 Base64 字符表；部分外部生成的 LMDB
        # 可能省略末尾填充，因此在解码前按需补齐。
        encoded_image += b"=" * (-len(encoded_image) % 4)
        try:
            image_bytes = base64.urlsafe_b64decode(encoded_image)
            with Image.open(BytesIO(image_bytes)) as opened_image:
                opened_image.load()
                image: Any = opened_image.convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"could not decode image {self.source}:{image_id} from {self.images_path}"
            ) from exc

        if self.image_transform is not None:
            image = self.image_transform(image)

        text_uid = make_uid(self.source, text_id)
        image_uid = make_uid(self.source, image_id)
        return {
            "image": image,
            "text": raw_text,
            "source": self.source,
            "text_id": text_id,
            "image_id": image_id,
            "text_uid": text_uid,
            "image_uid": image_uid,
        }


class MultiSourceDataset(Dataset):
    """拼接多个来源数据集，同时完整保留各自的正样本关系图。"""

    def __init__(self, datasets: Sequence[LMDBPairDataset]) -> None:
        super().__init__()
        if not datasets:
            raise ValueError("at least one source dataset is required")
        self.datasets = tuple(datasets)
        self.sources = tuple(dataset.source for dataset in self.datasets)
        if len(set(self.sources)) != len(self.sources):
            raise ValueError("source names must be unique within MultiSourceDataset")
        self.source_lengths = tuple(len(dataset) for dataset in self.datasets)
        if any(length <= 0 for length in self.source_lengths):
            raise ValueError("source datasets must be non-empty")

        cumulative = 0
        offsets: list[int] = []
        ends: list[int] = []
        for length in self.source_lengths:
            offsets.append(cumulative)
            cumulative += length
            ends.append(cumulative)
        self.source_offsets = tuple(offsets)
        self._cumulative_ends = tuple(ends)
        self.text_to_images, self.image_to_texts = merge_positive_relations(
            self.datasets
        )
        self.image_to_text = self.image_to_texts

    def __len__(self) -> int:
        return self._cumulative_ends[-1]

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = bisect_right(self._cumulative_ends, index)
        local_index = index - self.source_offsets[source_index]
        return self.datasets[source_index][local_index]

    def source_index_for_global_index(self, index: int) -> int:
        """返回拼接后全局索引所在的来源数据集序号。"""

        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self._cumulative_ends, index)


class TemperatureBalancedSampler(Sampler[int]):
    """按 ``p(source) ∝ sqrt(n)`` 对数据来源进行确定性温度平衡采样。

    使用最大余数法为每个来源分配精确的 epoch 样本数。来源内部先无放回打乱，
    用尽全部样本后再重新打乱，从而在需要过采样时生成后续样本。
    """

    def __init__(
        self,
        data_source: MultiSourceDataset | Mapping[str, int] | Sequence[int],
        *,
        epoch_size: int = DEFAULT_EPOCH_SIZE,
        seed: int = DEFAULT_SEED,
        exponent: float = 0.5,
    ) -> None:
        if isinstance(data_source, MultiSourceDataset):
            self.source_names = data_source.sources
            self.source_lengths = data_source.source_lengths
            self.source_offsets = data_source.source_offsets
        elif isinstance(data_source, Mapping):
            self.source_names = tuple(str(name) for name in data_source)
            self.source_lengths = tuple(int(value) for value in data_source.values())
            self.source_offsets = self._make_offsets(self.source_lengths)
        else:
            self.source_lengths = tuple(int(value) for value in data_source)
            self.source_names = tuple(str(index) for index in range(len(self.source_lengths)))
            self.source_offsets = self._make_offsets(self.source_lengths)

        if not self.source_lengths or any(length <= 0 for length in self.source_lengths):
            raise ValueError("all source lengths must be positive")
        self.epoch_size = int(epoch_size)
        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        self.seed = int(seed)
        self.exponent = float(exponent)
        if self.exponent <= 0:
            raise ValueError("exponent must be positive")
        self.epoch = 0
        self.position = 0
        self.source_probabilities = self._probabilities()
        self.source_counts = self._allocate_counts()

    @staticmethod
    def _make_offsets(lengths: Sequence[int]) -> tuple[int, ...]:
        offsets: list[int] = []
        total = 0
        for length in lengths:
            offsets.append(total)
            total += length
        return tuple(offsets)

    def _probabilities(self) -> tuple[float, ...]:
        weights = tuple(length**self.exponent for length in self.source_lengths)
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def _allocate_counts(self) -> tuple[int, ...]:
        raw_counts = [probability * self.epoch_size for probability in self.source_probabilities]
        counts = [int(value) for value in raw_counts]
        remainder = self.epoch_size - sum(counts)
        fractional_order = sorted(
            range(len(counts)),
            key=lambda index: (-(raw_counts[index] - counts[index]), index),
        )
        for index in fractional_order[:remainder]:
            counts[index] += 1
        return tuple(counts)

    def _indices_for_epoch(self) -> list[int]:
        # 使用较大的奇数乘数，让相邻 epoch 的伪随机序列充分分离。
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        sampled: list[int] = []
        for offset, source_length, requested in zip(
            self.source_offsets, self.source_lengths, self.source_counts
        ):
            remaining = requested
            while remaining:
                permutation = list(range(source_length))
                rng.shuffle(permutation)
                take = min(remaining, source_length)
                sampled.extend(offset + local_index for local_index in permutation[:take])
                remaining -= take
        rng.shuffle(sampled)
        if len(sampled) != self.epoch_size:
            raise RuntimeError("sampler generated an unexpected number of indices")
        return sampled

    def __iter__(self):
        indices = self._indices_for_epoch()
        start = min(self.position, self.epoch_size)
        for position in range(start, self.epoch_size):
            yield indices[position]

    def __len__(self) -> int:
        return self.epoch_size

    @property
    def remaining(self) -> int:
        return max(0, self.epoch_size - self.position)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.position = 0

    def advance(self, consumed_samples: int) -> None:
        """记录训练循环已经实际收到的样本数。

        DataLoader worker 可能预取远超当前优化批次的采样索引，因此
        ``__iter__`` 不能自行推进 checkpoint 游标；训练循环只有在真正收到
        一个 batch 后才调用本方法。
        """

        consumed_samples = int(consumed_samples)
        if consumed_samples < 0:
            raise ValueError("consumed_samples must be non-negative")
        next_position = self.position + consumed_samples
        if next_position > self.epoch_size:
            raise ValueError(
                f"sampler position would exceed epoch_size: {next_position} > "
                f"{self.epoch_size}"
            )
        self.position = next_position

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "position": self.position,
            "seed": self.seed,
            "epoch_size": self.epoch_size,
            "exponent": self.exponent,
            "source_names": self.source_names,
            "source_lengths": self.source_lengths,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "seed": self.seed,
            "epoch_size": self.epoch_size,
            "exponent": self.exponent,
            "source_names": self.source_names,
            "source_lengths": self.source_lengths,
        }
        for key, expected_value in expected.items():
            loaded_value = state.get(key)
            if isinstance(expected_value, tuple) and loaded_value is not None:
                loaded_value = tuple(loaded_value)
            if loaded_value != expected_value:
                raise ValueError(
                    f"sampler checkpoint {key}={loaded_value!r} does not match "
                    f"current {key}={expected_value!r}"
                )
        epoch = int(state["epoch"])
        position = int(state["position"])
        if epoch < 0 or position < 0 or position > self.epoch_size:
            raise ValueError("invalid sampler epoch/position in checkpoint")
        self.epoch = epoch
        self.position = position


def build_positive_mask(
    text_uids: Sequence[str],
    image_uids: Sequence[str],
    text_to_images: Mapping[str, Iterable[str]],
    image_to_texts: Mapping[str, Iterable[str]] | None = None,
    *,
    require_positive: bool = True,
) -> torch.Tensor:
    """依据完整数据集关系图构造当前 batch 的多正样本布尔矩阵。"""

    mask = torch.zeros((len(text_uids), len(image_uids)), dtype=torch.bool)
    image_to_texts = image_to_texts or {}
    for text_index, text_uid in enumerate(text_uids):
        text_uid = str(text_uid)
        known_images = text_to_images.get(text_uid, ())
        for image_index, image_uid in enumerate(image_uids):
            image_uid = str(image_uid)
            if image_uid in known_images or text_uid in image_to_texts.get(image_uid, ()):
                mask[text_index, image_index] = True

    if require_positive and mask.numel():
        missing_texts = (~mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        missing_images = (~mask.any(dim=0)).nonzero(as_tuple=False).flatten().tolist()
        if missing_texts or missing_images:
            raise ValueError(
                "batch relation graph has anchors without a positive: "
                f"text rows={missing_texts}, image columns={missing_images}"
            )
    return mask


class MultiPositiveCollator:
    """处理成对图文样本，并附加当前 batch 的完整多正样本矩阵。"""

    def __init__(
        self,
        processor: Callable[..., Mapping[str, Any]],
        text_to_images: Mapping[str, Iterable[str]],
        image_to_texts: Mapping[str, Iterable[str]],
        *,
        max_length: int = 52,
        padding: bool | str = "max_length",
    ) -> None:
        self.processor = processor
        self.text_to_images = text_to_images
        self.image_to_texts = image_to_texts
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        self.padding = padding

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty batch")
        images = [sample["image"] for sample in samples]
        texts = [str(sample["text"]) for sample in samples]
        text_uids = [str(sample["text_uid"]) for sample in samples]
        image_uids = [str(sample["image_uid"]) for sample in samples]

        encoded = self.processor(
            images=images,
            text=texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("processor must return a mapping of model inputs")
        batch = dict(encoded)
        required = {"pixel_values", "input_ids", "attention_mask"}
        missing = sorted(required.difference(batch))
        if missing:
            raise KeyError(f"processor output is missing required fields: {missing}")

        batch.update(
            {
                "positive_mask": build_positive_mask(
                    text_uids,
                    image_uids,
                    self.text_to_images,
                    self.image_to_texts,
                    require_positive=True,
                ),
                "text_uids": text_uids,
                "image_uids": image_uids,
                "texts": texts,
                "sources": [str(sample["source"]) for sample in samples],
                "text_ids": [sample["text_id"] for sample in samples],
                "image_ids": [sample["image_id"] for sample in samples],
            }
        )
        return batch


__all__ = [
    "DEFAULT_EPOCH_SIZE",
    "DEFAULT_SEED",
    "LMDBPairDataset",
    "MultiPositiveCollator",
    "MultiSourceDataset",
    "TemperatureBalancedSampler",
    "build_positive_mask",
    "build_positive_relations",
    "make_uid",
    "merge_positive_relations",
]
