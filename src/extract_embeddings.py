"""
全量向量抽取脚本。

脚本读取样本清单，把每条样本的文本和图片分别编码成向量矩阵。两个矩阵都保持
“第几行对应第几条样本”的关系，后续检索拿到行号后即可回查原始元信息。
"""

"""引入命令行参数解析工具。"""
import argparse
"""引入结构化文本工具，用来写图片错误日志。"""
import json
"""引入系统退出工具，用于把主函数返回值作为进程退出码。"""
import sys
"""引入路径对象，方便拼接输入输出路径。"""
from pathlib import Path
"""引入类型标注，帮助读者理解字典和列表内容。"""
from typing import Any, Dict, List, Mapping

"""引入矩阵计算库，用来创建和写入向量矩阵。"""
import numpy as np
"""引入深度学习框架，用来运行图文模型。"""
import torch
"""引入向量归一化函数。"""
import torch.nn.functional as F
"""引入图片读取库和图片读取配置。"""
from PIL import Image, ImageFile
"""引入进度条工具，用来显示长任务进度。"""
from tqdm import tqdm
"""引入中文图文模型和对应处理器。"""
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

"""兼容命令行直接运行和作为 ``src`` 包导入两种方式。"""
try:
    from model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
        build_compatibility_metadata,
        compute_data_manifest_hash,
        compute_model_fingerprint,
    )
    from utils import Timer, ensure_dir, item_text, load_json, read_jsonl, resolve_path, save_json
except ImportError:
    from src.model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
        build_compatibility_metadata,
        compute_data_manifest_hash,
        compute_model_fingerprint,
    )
    from src.utils import Timer, ensure_dir, item_text, load_json, read_jsonl, resolve_path, save_json

"""
允许读取部分截断但仍可解析的图片。

全量数据里偶尔会出现不完整图片，如果直接报错会中断整个任务。
"""
ImageFile.LOAD_TRUNCATED_IMAGES = True

 
def parse_args() -> argparse.Namespace:
    """
    读取向量抽取参数。

    这里可以指定输入清单、项目根目录、输出目录、批次大小、运行设备、是否断点续跑、
    是否按图片路径去重等配置。
    """
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="抽取文本向量和图片向量")
    """声明样本清单路径。"""
    parser.add_argument("--items", default="data/processed/demo/items.jsonl")
    """声明项目根目录，用来解析相对图片路径。"""
    parser.add_argument("--project_root", default=".")
    """声明向量和元信息输出目录。"""
    parser.add_argument("--output_dir", default="outputs/demo_full")
    """声明使用的中文图文模型名称或本地目录。"""
    parser.add_argument("--model_name", default="OFA-Sys/chinese-clip-vit-base-patch16")
    """声明最多读取多少条样本；为空时读取全量。"""
    parser.add_argument("--limit", type=int, default=None)
    """声明文本编码批次大小。"""
    parser.add_argument("--text_batch_size", type=int, default=256)
    """声明图片编码批次大小。"""
    parser.add_argument("--image_batch_size", type=int, default=64)
    """声明文本最大长度；留空时自动读取最终模型的 retrieval_config.json。"""
    parser.add_argument("--max_length", type=int, default=None)
    """声明运行设备。"""
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    """声明是否在显卡上启用半精度推理。"""
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    """声明是否从已有状态和矩阵文件继续运行。"""
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    """声明是否按图片路径去重后再抽取图片向量。"""
    parser.add_argument("--dedupe_images", action=argparse.BooleanOptionalAction, default=True)
    """声明每隔多少批次把矩阵和状态刷回磁盘。"""
    parser.add_argument("--flush_every_batches", type=int, default=10)
    """解析并返回命令行参数。"""
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    """
    选择运行设备。

    自动模式下优先使用显卡；如果没有显卡，就退回处理器。保留显式选项是为了远程排查时
    可以强制指定运行设备。
    """
    """如果请求自动选择设备，就根据显卡是否可用来决定。"""
    if requested == "auto":
        """显卡可用时使用显卡，否则使用处理器。"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """如果用户显式指定设备，就直接使用指定设备。"""
    return torch.device(requested)


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    把分词器或图像处理器生成的张量移动到指定设备。

    模型和输入必须在同一个设备上，否则推理时会报设备不一致。
    """
    """遍历输入字典，把每个张量移动到指定设备后重新组成字典。"""
    return {key: value.to(device) for key, value in batch.items()}


def unwrap_features(features) -> torch.Tensor:
    """
    从模型输出中抽出真正的向量张量。

    不同版本的模型库返回形态不完全一致，可能是张量、对象或元组；这里统一兼容。
    """
    """如果模型已经直接返回张量，就直接使用。"""
    if isinstance(features, torch.Tensor):
        """返回张量形式的向量。"""
        return features
    """按常见字段名依次尝试读取向量张量。"""
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        """读取当前字段值。"""
        value = getattr(features, attr, None)
        """如果当前字段是张量，就认为它是需要的向量。"""
        if isinstance(value, torch.Tensor):
            """返回找到的向量张量。"""
            return value
    """部分版本可能用元组或列表返回，第一个元素就是向量张量。"""
    if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
        """返回序列中的第一个张量。"""
        return features[0]
    """所有兼容路径都失败时，抛出类型错误。"""
    raise TypeError(f"无法从模型输出中找到向量张量，输出类型为 {type(features)!r}")


def normalize_features(features) -> np.ndarray:
    """
    归一化模型输出并转成矩阵。

    归一化后向量长度为一，后续内积检索即可等价表示余弦相似度。
    """
    """兼容不同返回格式，抽出真正的向量张量。"""
    features = unwrap_features(features)
    """对向量按最后一维做归一化。"""
    features = F.normalize(features, p=2, dim=-1)
    """把向量转到处理器内存，并转成索引库友好的矩阵格式。"""
    return features.detach().float().cpu().numpy().astype("float32")


def load_image(path: Path) -> Image.Image:
    """
    读取图片并转成三通道格式。

    如果图片读取失败，返回一张白底占位图。这样可以保证向量矩阵行数不乱，坏图位置后续
    通过错误日志排查。
    """
    """尝试打开图片并转换成三通道格式。"""
    try:
        """返回可被模型处理的图片对象。"""
        return Image.open(path).convert("RGB")
    except Exception:
        """读取失败时返回白色占位图，避免全量任务中断。"""
        return Image.new("RGB", (224, 224), color="white")


def load_retrieval_config(model_dir: str | Path) -> dict[str, Any]:
    """读取最终模型的检索配置；旧模型没有该文件时返回空配置。"""

    path = Path(model_dir) / "retrieval_config.json"
    if not path.is_file():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"检索配置根节点必须是对象：{path}")
    return payload


def resolve_max_length(requested: int | None, retrieval_config: Mapping[str, Any]) -> int:
    """优先采用模型随附的 max_length，并拒绝会改变向量语义的手工冲突值。"""

    configured = retrieval_config.get("max_length")
    if configured is not None:
        configured = int(configured)
        if configured <= 0:
            raise ValueError("retrieval_config.max_length 必须是正整数")
        if requested is not None and int(requested) != configured:
            raise MetadataMismatchError(
                f"命令行 max_length={requested} 与最终模型 retrieval_config.max_length={configured} 不一致"
            )
        return configured
    value = 256 if requested is None else int(requested)
    if value <= 0:
        raise ValueError("max_length 必须是正整数")
    return value


def _compact_items(items: List[dict]) -> List[dict]:
    """生成与向量矩阵逐行对齐的轻量展示元信息。"""

    compact_items: List[dict] = []
    for row_id, item in enumerate(items):
        compact_items.append(
            {
                "row_id": row_id,
                "item_id": item.get("item_id") or item.get("note_id") or str(row_id),
                "text": item_text(item),
                "image_id": item.get("image_id"),
                "image_path": item.get("image_path"),
                "source": item.get("source"),
                "split": item.get("split"),
                "text_id": item.get("text_id"),
                "title": item.get("title"),
                "content": item.get("content"),
                "tags": item.get("tags"),
            }
        )
    return compact_items


def _compatibility_field_status(payload: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """返回已存在和缺失的四个强制兼容字段。"""

    present = {field for field in REQUIRED_COMPATIBILITY_FIELDS if field in payload}
    return present, set(REQUIRED_COMPATIBILITY_FIELDS) - present


def _state_has_zero_progress(state: Mapping[str, Any]) -> bool:
    """仅当所有断点计数均为零且未完成时，才把旧状态视为可升级。"""

    if state.get("complete"):
        return False
    for field in ("text_done", "image_done", "image_unique_done"):
        try:
            if int(state.get(field, 0) or 0) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _assert_resume_files_exist(output_dir: Path, state: Mapping[str, Any]) -> None:
    """已有进度必须有对应矩阵，避免把缺失文件静默重建为未初始化内容。"""

    text_done = int(state.get("text_done", 0) or 0)
    image_done = max(
        int(state.get("image_done", 0) or 0),
        int(state.get("image_unique_done", 0) or 0),
    )
    if (text_done > 0 or state.get("complete")) and not (output_dir / "text_embeddings.npy").is_file():
        raise FileNotFoundError("断点状态已有文本进度，但 text_embeddings.npy 不存在")
    if (image_done > 0 or state.get("complete")) and not (output_dir / "image_embeddings.npy").is_file():
        raise FileNotFoundError("断点状态已有图片进度，但 image_embeddings.npy 不存在")


def prepare_output_metadata(
    items: List[dict],
    output_dir: Path,
    args: argparse.Namespace,
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """校验或初始化抽取状态；任何不兼容都发生在矩阵写入之前。

    老版本产物没有四个兼容字段时，仅允许进度明确为零的状态就地升级。已经写过任意
    向量的老产物无法证明模型和数据来源，必须换新目录重新抽取。
    """

    state_path = output_dir / "extract_state.json"
    meta_path = output_dir / "item_meta.json"
    matrix_paths = (output_dir / "text_embeddings.npy", output_dir / "image_embeddings.npy")

    if not args.resume:
        existing = [path for path in output_dir.iterdir()]
        if existing:
            names = ", ".join(sorted(path.name for path in existing))
            raise FileExistsError(f"输出目录非空，拒绝覆盖旧输出：{output_dir}（{names}）")

    raw_state = load_json(state_path, default=None)
    raw_meta = load_json(meta_path, default=None)
    if raw_state is not None and not isinstance(raw_state, dict):
        raise ValueError(f"断点状态根节点必须是对象：{state_path}")
    if raw_meta is not None and not isinstance(raw_meta, dict):
        raise ValueError(f"样本元信息根节点必须是对象：{meta_path}")

    state: dict[str, Any] = dict(raw_state or {})
    vector_files_exist = any(path.exists() for path in matrix_paths)
    if args.resume and raw_state is None and raw_meta is None and any(output_dir.iterdir()):
        raise FileExistsError("输出目录已有无法识别的旧产物，拒绝覆盖；请改用新的 output_dir")
    if raw_state is None and vector_files_exist:
        raise MetadataMismatchError("存在旧向量矩阵但缺少 extract_state.json，无法确认零进度，禁止升级")
    zero_progress = _state_has_zero_progress(state) if raw_state is not None else not vector_files_exist

    if raw_state is not None:
        present, missing = _compatibility_field_status(state)
        if present and missing:
            assert_metadata_compatible(state, compatibility, context="抽取断点")
        elif present:
            assert_metadata_compatible(state, compatibility, context="抽取断点")
        elif not zero_progress:
            raise MetadataMismatchError("旧版抽取断点已有进度且缺少四字段元数据，禁止升级或续跑")
        _assert_resume_files_exist(output_dir, state)
        if not state.get("complete") and any(
            (output_dir / name).exists() for name in ("text.index", "image.index", "index_meta.json")
        ):
            raise MetadataMismatchError("未完成的抽取目录中已有索引产物，拒绝续写以免索引与向量错配")

    if raw_meta is not None:
        if "items" not in raw_meta or not isinstance(raw_meta["items"], list):
            raise ValueError(f"样本元信息缺少 items 列表：{meta_path}")
        if int(raw_meta.get("count", len(raw_meta["items"]))) != len(items) or len(raw_meta["items"]) != len(items):
            raise MetadataMismatchError("item_meta.json 的样本数与本次 items 文件不一致")
        present, missing = _compatibility_field_status(raw_meta)
        if present and missing:
            assert_metadata_compatible(raw_meta, compatibility, context="样本元信息")
        elif present:
            assert_metadata_compatible(raw_meta, compatibility, context="样本元信息")
        elif not zero_progress:
            raise MetadataMismatchError("旧版 item_meta 已对应非零抽取进度，缺少四字段元数据，禁止升级")
        meta = dict(raw_meta)
    else:
        if raw_state is not None and not zero_progress and state:
            raise MetadataMismatchError("已有非零抽取进度但缺少 item_meta.json，禁止继续")
        meta = {
            "count": len(items),
            "model_name": args.model_name,
            "items_path": args.items,
            "items": _compact_items(items),
        }

    # 所有检查均已通过，此后才允许给零进度老产物补写新格式元数据。
    meta.update(dict(compatibility))
    meta["count"] = len(items)
    meta["max_length"] = int(args.max_length)
    save_json(meta, meta_path)

    state.update(dict(compatibility))
    state.update(
        {
            "items": args.items,
            "count": len(items),
            "model_name": args.model_name,
            "projection_dim": int(compatibility["dimension"]),
            "max_length": int(args.max_length),
            "text_batch_size": args.text_batch_size,
            "image_batch_size": args.image_batch_size,
            "dedupe_images": args.dedupe_images,
        }
    )
    save_json(state, state_path)
    return state


def write_meta(
    items: List[dict],
    output_dir: Path,
    args: argparse.Namespace,
    compatibility: Mapping[str, Any] | None = None,
) -> None:
    """保存样本元信息；兼容旧调用，但新抽取应通过 prepare_output_metadata。"""

    target = output_dir / "item_meta.json"
    if target.exists():
        raise FileExistsError("item_meta.json 已存在，拒绝覆盖；断点续跑请使用 prepare_output_metadata")
    payload = {
        "count": len(items),
        "model_name": args.model_name,
        "items_path": args.items,
        "items": _compact_items(items),
    }
    if compatibility:
        payload.update(dict(compatibility))
        payload["max_length"] = int(args.max_length)
    save_json(payload, target)


def open_embedding_array(path: Path, shape: tuple[int, int], resume: bool) -> np.memmap:
    """
    打开或创建向量矩阵文件。

    使用磁盘映射方式写入，可以避免把完整矩阵一直放在内存里；断点续跑时也能直接在原文件
    上继续补写。
    """
    """如果允许断点续跑且矩阵文件已经存在，就尝试打开旧文件。"""
    if resume and path.exists():
        """以可读写磁盘映射方式打开已有矩阵。"""
        arr = np.load(path, mmap_mode="r+")
        """检查已有矩阵形状是否和本次任务一致。"""
        if arr.shape != shape:
            """形状不一致说明不能安全续跑，直接报错。"""
            raise ValueError(f"{path} 的形状是 {arr.shape}，但本次期望形状是 {shape}")
        """返回可继续写入的已有矩阵。"""
        return arr
    """如果不能续跑或文件不存在，就创建新的磁盘映射矩阵。"""
    return np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=shape)


def infer_projection_dim(model: ChineseCLIPModel) -> int:
    """
    推断模型输出向量维度。

    维度通常在模型配置里；如果配置字段缺失，就使用当前模型常见维度兜底。
    """
    """优先从模型主配置里读取投影维度。"""
    dim = getattr(model.config, "projection_dim", None)
    """如果主配置没有，就尝试从文本子配置里读取。"""
    if dim is None:
        """读取文本子配置里的投影维度。"""
        dim = getattr(getattr(model.config, "text_config", None), "projection_dim", None)
    """返回整数维度；仍然缺失时使用默认维度。"""
    return int(dim or 512)


def extract_text_embeddings(
    items: List[dict],
    output_dir: Path,
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    args: argparse.Namespace,
    dim: int,
    state: dict,
) -> None:
    """
    抽取文本向量。

    文本向量矩阵的行数等于样本数，列数等于模型输出维度。每完成若干批次就写回矩阵和
    状态文件，远程断连或进程中断后可以继续跑。
    """
    """拼出文本向量矩阵路径。"""
    text_path = output_dir / "text_embeddings.npy"
    """打开或创建文本向量矩阵。"""
    text_embeddings = open_embedding_array(text_path, (len(items), dim), args.resume)
    """读取断点位置；不续跑时从零开始。"""
    start = int(state.get("text_done", 0)) if args.resume else 0
    """如果断点已经覆盖全部样本，说明文本向量已经完成。"""
    if start >= len(items):
        """输出已完成提示。"""
        print(f"文本向量已完成：{start}/{len(items)}")
        """结束文本向量抽取。"""
        return

    """创建计时器，用于进度条展示耗时。"""
    timer = Timer()
    """创建文本向量抽取进度条。"""
    pbar = tqdm(range(start, len(items), args.text_batch_size), desc="文本向量", dynamic_ncols=True)
    """按批次遍历文本样本。"""
    for batch_idx, batch_start in enumerate(pbar, start=1):
        """计算当前批次结束位置。"""
        batch_end = min(batch_start + args.text_batch_size, len(items))
        """取出当前批次文本。"""
        texts = [item_text(item) for item in items[batch_start:batch_end]]
        """把文本批次转成模型输入张量。"""
        encoded = processor(
            text=texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        """把输入张量移动到模型所在设备。"""
        encoded = move_to_device(encoded, device)
        """关闭梯度计算，只做推理。"""
        with torch.inference_mode():
            """显卡上允许半精度时，用半精度推理减少显存占用。"""
            if device.type == "cuda" and args.fp16:
                """进入显卡半精度上下文。"""
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    """调用文本编码接口得到当前批次文本向量。"""
                    features = model.get_text_features(**encoded)
            else:
                """非半精度场景下直接调用文本编码接口。"""
                features = model.get_text_features(**encoded)
        """归一化当前批次向量并写入矩阵对应行。"""
        text_embeddings[batch_start:batch_end] = normalize_features(features)

        """按配置间隔刷盘，或者在最后一个批次强制刷盘。"""
        if batch_idx % args.flush_every_batches == 0 or batch_end == len(items):
            """把内存里的矩阵变更写回磁盘。"""
            text_embeddings.flush()
            """更新文本断点位置。"""
            state["text_done"] = batch_end
            """保存最新断点状态。"""
            save_json(state, output_dir / "extract_state.json")
        """更新进度条附加信息。"""
        pbar.set_postfix(done=batch_end, elapsed=timer.elapsed_text())


def build_image_groups(items: List[dict]) -> tuple[List[str], Dict[str, List[int]]]:
    """
    按图片路径对样本分组。

    多条文本可能共用同一张图片；图片只需要编码一次，再把同一个图片向量回填到所有同图样本。
    """
    """准备图片路径到样本行号列表的映射。"""
    groups: Dict[str, List[int]] = {}
    """准备唯一图片路径的出现顺序。"""
    order: List[str] = []
    """遍历所有样本。"""
    for row_id, item in enumerate(items):
        """读取当前样本的图片路径，缺失时用空字符串占位。"""
        image_path = str(item.get("image_path") or "")
        """如果这是第一次看到该图片路径，就初始化分组。"""
        if image_path not in groups:
            """创建当前图片路径对应的样本行号列表。"""
            groups[image_path] = []
            """记录唯一图片路径的顺序。"""
            order.append(image_path)
        """把当前样本行号加入该图片路径分组。"""
        groups[image_path].append(row_id)
    """返回唯一图片路径列表和路径到行号的映射。"""
    return order, groups


def extract_image_embeddings(
    items: List[dict],
    output_dir: Path,
    processor: ChineseCLIPProcessor,
    model: ChineseCLIPModel,
    device: torch.device,
    args: argparse.Namespace,
    dim: int,
    state: dict,
) -> None:
    """
    抽取图片向量。

    最终图片向量矩阵仍然保持每条样本一行。开启去重时，脚本先对唯一图片编码，再把结果
    回填到所有使用该图片的样本行。
    """
    """拼出图片向量矩阵路径。"""
    image_path = output_dir / "image_embeddings.npy"
    """打开或创建图片向量矩阵。"""
    image_embeddings = open_embedding_array(image_path, (len(items), dim), args.resume)
    """解析项目根目录，用于把相对图片路径转成真实路径。"""
    project_root = Path(args.project_root).resolve()
    """创建计时器，用于进度条展示耗时。"""
    timer = Timer()

    """如果启用图片去重，就按唯一图片抽向量再回填。"""
    if args.dedupe_images:
        """构建唯一图片路径列表和图片路径到样本行号的映射。"""
        unique_paths, groups = build_image_groups(items)
        """读取唯一图片断点位置；不续跑时从零开始。"""
        start = int(state.get("image_unique_done", 0)) if args.resume else 0
        """如果断点已经覆盖全部唯一图片，说明图片向量已经完成。"""
        if start >= len(unique_paths):
            """输出已完成提示。"""
            print(f"去重图片向量已完成：{start}/{len(unique_paths)} 张唯一图片")
            """结束图片向量抽取。"""
            return

        """创建去重图片向量抽取进度条。"""
        pbar = tqdm(range(start, len(unique_paths), args.image_batch_size), desc="去重图片向量", dynamic_ncols=True)
        """打开图片错误日志文件，追加记录缺失或读取失败情况。"""
        error_log = (output_dir / "image_errors.jsonl").open("a", encoding="utf-8")
        """确保无论中途是否报错，错误日志文件都会关闭。"""
        try:
            """按批次遍历唯一图片路径。"""
            for batch_idx, batch_start in enumerate(pbar, start=1):
                """计算当前批次结束位置。"""
                batch_end = min(batch_start + args.image_batch_size, len(unique_paths))
                """取出当前批次图片路径。"""
                paths = unique_paths[batch_start:batch_end]
                """准备当前批次的图片对象列表。"""
                images = []
                """逐张读取当前批次图片。"""
                for raw_path in paths:
                    """把样本中的图片路径解析成真实文件路径。"""
                    abs_path = resolve_path(project_root, raw_path)
                    """如果图片文件不存在，就记录错误日志。"""
                    if not abs_path.exists():
                        """写入缺失图片记录。"""
                        error_log.write(json.dumps({"image_path": raw_path, "error": "文件不存在"}, ensure_ascii=False) + "\n")
                    """读取图片；读取失败时函数内部会返回白图占位。"""
                    images.append(load_image(abs_path))

                """把图片批次转成模型输入张量。"""
                encoded = processor(images=images, return_tensors="pt")
                """把输入张量移动到模型所在设备。"""
                encoded = move_to_device(encoded, device)
                """关闭梯度计算，只做推理。"""
                with torch.inference_mode():
                    """显卡上允许半精度时，用半精度推理减少显存占用。"""
                    if device.type == "cuda" and args.fp16:
                        """进入显卡半精度上下文。"""
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            """调用图片编码接口得到当前批次图片向量。"""
                            features = model.get_image_features(**encoded)
                    else:
                        """非半精度场景下直接调用图片编码接口。"""
                        features = model.get_image_features(**encoded)
                """归一化当前批次图片向量。"""
                batch_features = normalize_features(features)

                """把唯一图片向量回填到所有使用该图片的样本行。"""
                for local_idx, raw_path in enumerate(paths):
                    """找到当前唯一图片对应的所有样本行号。"""
                    rows = groups[raw_path]
                    """把同一个图片向量写到这些样本行中。"""
                    image_embeddings[rows] = batch_features[local_idx]

                """按配置间隔刷盘，或者在最后一个批次强制刷盘。"""
                if batch_idx % args.flush_every_batches == 0 or batch_end == len(unique_paths):
                    """把内存里的矩阵变更写回磁盘。"""
                    image_embeddings.flush()
                    """把错误日志缓冲内容写回磁盘。"""
                    error_log.flush()
                    """更新唯一图片断点位置。"""
                    state["image_unique_done"] = batch_end
                    """记录唯一图片总数。"""
                    state["image_unique_total"] = len(unique_paths)
                    """保存最新断点状态。"""
                    save_json(state, output_dir / "extract_state.json")
                """更新进度条附加信息。"""
                pbar.set_postfix(done=batch_end, total=len(unique_paths), elapsed=timer.elapsed_text())
        finally:
            """关闭图片错误日志文件。"""
            error_log.close()
    else:
        """不启用图片去重时，按样本行逐条抽取图片向量。"""
        start = int(state.get("image_done", 0)) if args.resume else 0
        """如果断点已经覆盖全部样本，说明图片向量已经完成。"""
        if start >= len(items):
            """输出已完成提示。"""
            print(f"图片向量已完成：{start}/{len(items)}")
            """结束图片向量抽取。"""
            return
        """创建逐样本图片向量抽取进度条。"""
        pbar = tqdm(range(start, len(items), args.image_batch_size), desc="图片向量", dynamic_ncols=True)
        """按批次遍历样本图片。"""
        for batch_idx, batch_start in enumerate(pbar, start=1):
            """计算当前批次结束位置。"""
            batch_end = min(batch_start + args.image_batch_size, len(items))
            """读取当前批次图片。"""
            images = [load_image(resolve_path(project_root, item.get("image_path") or "")) for item in items[batch_start:batch_end]]
            """把图片批次转成模型输入并移动到设备。"""
            encoded = move_to_device(processor(images=images, return_tensors="pt"), device)
            """关闭梯度计算，只做推理。"""
            with torch.inference_mode():
                """调用图片编码接口得到当前批次图片向量。"""
                features = model.get_image_features(**encoded)
            """归一化当前批次向量并写入矩阵对应行。"""
            image_embeddings[batch_start:batch_end] = normalize_features(features)
            """按配置间隔刷盘，或者在最后一个批次强制刷盘。"""
            if batch_idx % args.flush_every_batches == 0 or batch_end == len(items):
                """把内存里的矩阵变更写回磁盘。"""
                image_embeddings.flush()
                """更新逐样本图片断点位置。"""
                state["image_done"] = batch_end
                """保存最新断点状态。"""
                save_json(state, output_dir / "extract_state.json")
            """更新进度条附加信息。"""
            pbar.set_postfix(done=batch_end, elapsed=timer.elapsed_text())


def main() -> int:
    """
    主入口。

    负责读取样本、加载模型、写元信息、恢复状态，并依次完成文本向量和图片向量抽取。
    """
    """读取命令行参数。"""
    args = parse_args()
    """确保输出目录存在。"""
    output_dir = ensure_dir(args.output_dir)
    """读取最终模型随附的检索参数，并自动采用训练/导出时的文本长度。"""
    retrieval_config = load_retrieval_config(args.model_name)
    args.max_length = resolve_max_length(args.max_length, retrieval_config)
    """读取样本清单；如果设置了上限，就只读取前若干条。"""
    items = read_jsonl(args.items, limit=args.limit)
    """如果没有读到任何样本，直接报错。"""
    if not items:
        """抛出空数据错误。"""
        raise ValueError(f"没有从 {args.items} 读取到任何样本")

    """选择模型运行设备。"""
    device = choose_device(args.device)
    """输出样本数量。"""
    print(f"已读取样本数：{len(items)}")
    """输出运行设备。"""
    print(f"运行设备：{device}")
    """输出向量保存目录。"""
    print(f"输出目录：{output_dir}")

    """加载模型处理器。"""
    processor = ChineseCLIPProcessor.from_pretrained(args.model_name)
    """加载图文模型。"""
    model = ChineseCLIPModel.from_pretrained(args.model_name)
    """切换到推理模式并移动到指定设备。"""
    model.eval().to(device)
    """推断模型输出向量维度。"""
    dim = infer_projection_dim(model)
    """最终模型配置声明的维度必须和实际加载结果一致。"""
    configured_dim = retrieval_config.get("dimension")
    if configured_dim is not None and int(configured_dim) != dim:
        raise MetadataMismatchError(
            f"retrieval_config.dimension={configured_dim} 与模型实际投影维度 {dim} 不一致"
        )
    """当前索引固定使用归一化向量；其他方式不能复用这条检索链路。"""
    normalization = str(retrieval_config.get("normalization", "l2")).lower()
    if normalization != "l2":
        raise MetadataMismatchError(f"当前抽取流程只支持 l2 归一化，模型配置为 {normalization!r}")
    """输出向量维度。"""
    print(f"向量维度：{dim}")
    """输出自动采用的文本最大长度。"""
    print(f"文本最大长度：{args.max_length}")

    """根据最终落盘模型内容和原始 items 文件字节计算本次唯一兼容契约。"""
    model_dir = Path(args.model_name)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "无法为远程模型名称计算稳定内容指纹；请把 --model_name 指向最终导出的本地模型目录"
        )
    compatibility = build_compatibility_metadata(
        model_fingerprint=compute_model_fingerprint(model_dir),
        data_manifest_hash=compute_data_manifest_hash(Path(args.items)),
        dimension=dim,
        normalization=normalization,
    )
    """在第一次写矩阵前完成断点和样本元信息的四字段 fail-fast 校验。"""
    state = prepare_output_metadata(items, output_dir, args, compatibility)

    """抽取文本向量。"""
    extract_text_embeddings(items, output_dir, processor, model, device, args, dim, state)
    """抽取图片向量。"""
    extract_image_embeddings(items, output_dir, processor, model, device, args, dim, state)
    """标记本次向量抽取任务已完整完成。"""
    state["complete"] = True
    """保存最终状态。"""
    save_json(state, output_dir / "extract_state.json")
    """输出完成提示。"""
    print("向量抽取已完成")
    """正常结束。"""
    return 0


"""只有当本文件作为脚本直接运行时，才执行主入口。"""
if __name__ == "__main__":
    """把主函数返回值作为进程退出码。"""
    sys.exit(main())
