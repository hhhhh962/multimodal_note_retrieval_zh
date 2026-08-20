"""
四路召回搜索流水线。

这个模块负责把用户输入的文本和图片编码成 Chinese-CLIP 查询向量，并在文本索引和图片索引中
分别召回候选。文本查询和图片查询最多形成四路相似度分数，最后按固定 late fusion 公式输出图文结果。
"""

"""引入系统环境工具，用来读取模型目录等运行配置。"""
import os
"""引入告警工具，用于提示仍可读取但没有新签名字段的老 baseline。"""
import warnings
"""引入结构化文本工具，用来调试和序列化结果。"""
import json
"""引入路径对象，方便解析项目内图片和索引路径。"""
from pathlib import Path
"""引入类型标注，帮助读者理解输入输出。"""
from typing import Any, Dict, List, Mapping, Optional

"""引入向量索引库，用来读取和检索 FAISS 索引。"""
import faiss
"""引入矩阵计算库，用来整理 FAISS 输入向量。"""
import numpy as np
"""引入图片读取库，用来处理图片查询。"""
from PIL import Image
"""引入深度学习框架，用来执行 Chinese-CLIP 推理。"""
import torch
"""引入向量归一化函数，使内积相似度等价于余弦相似度。"""
import torch.nn.functional as F
"""引入中文图文模型和预处理器。"""
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

"""兼容命令行直接运行和 app.py 包导入两种路径。"""
try:
    """命令行从 src 目录脚本启动时使用这个导入路径。"""
    from model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
        compute_model_fingerprint,
    )
    from utils import load_json, resolve_path
    from muge_aggregation import aggregate_results, load_muge_mapping
except ImportError:
    """app.py 从项目根目录导入 src.search_pipeline 时使用这个导入路径。"""
    from src.model_fingerprint import (
        REQUIRED_COMPATIBILITY_FIELDS,
        MetadataMismatchError,
        assert_metadata_compatible,
        compute_model_fingerprint,
    )
    from src.utils import load_json, resolve_path
    from src.muge_aggregation import aggregate_results, load_muge_mapping


"""记录当前正式部署使用的微调模型和全量索引默认路径。"""
DEFAULT_MODEL_DIR = "outputs/finetune_lora/exported_model"
DEFAULT_EMBEDDING_DIR = "outputs/finetuned_full"


class SearchPipeline:
    """
    常驻搜索流水线。

    初始化阶段加载模型、处理器、文本索引、图片索引和元信息；搜索阶段只编码本次 query 并检索索引。
    这样 Demo 连续查询时不用反复冷启动大模型和大索引。
    """

    def __init__(
        self,
        embedding_dir: str | Path = DEFAULT_EMBEDDING_DIR,
        model_name: str | None = None,
        project_root: str | Path = ".",
        device: str = "auto",
        use_gpu_faiss: bool = True,
        max_length: int | None = None,
        fp16: bool = True,
        ef_search: int | None = None,
    ) -> None:
        """
        初始化搜索流水线。

        model_name 可以传 Hugging Face 模型名，也可以传本地缓存快照目录；device 为 auto 时优先使用显卡，
        当前无卡环境会自动退回处理器。
        """
        """记录项目根目录，用来解析相对图片路径。"""
        self.project_root = Path(project_root)
        """记录索引和元信息所在目录。"""
        self.embedding_dir = Path(embedding_dir)
        """记录是否在显卡上使用半精度查询编码。"""
        self.fp16 = fp16
        """确定模型路径；优先使用调用方传入值，再读环境变量，最后使用项目内合并后的微调模型。"""
        self.model_name = (
            model_name
            or os.environ.get("MODEL_DIR")
            or os.environ.get("MODEL_NAME")
            or str(self.project_root / DEFAULT_MODEL_DIR)
        )
        """最终模型随附配置优先，保证查询编码与向量抽取都使用 max_length=52。"""
        self.retrieval_config = self.load_retrieval_config(self.model_name)
        self.max_length = self.resolve_max_length(max_length, self.retrieval_config)
        """选择模型推理设备。"""
        self.device = self.choose_device(device)
        """保存 FAISS 显卡资源对象，避免显卡索引依赖的资源被回收。"""
        self.gpu_resources = []
        """加载样本元信息，搜索结果需要通过 row_id 回查文本和图片。"""
        self.meta = self.load_meta(self.embedding_dir / "item_meta.json")
        """取出和向量矩阵行号一一对应的样本列表。"""
        self.items = self.meta["items"]
        """加载 MUGE 一文多图聚合映射（row_id -> doc_id）。"""
        self.muge_mapping = load_muge_mapping(self.embedding_dir / "muge_mapping.json")
        self.row_to_text_id = self.muge_mapping["row_to_text_id"]
        """索引元信息在新格式下是强制契约；老 baseline 可缺失。"""
        self.index_meta = load_json(self.embedding_dir / "index_meta.json", default=None)
        self.metadata_mode = self.detect_metadata_mode(self.meta, self.index_meta)
        """解析 HNSW 查询深度；显式参数优先于索引元数据。"""
        self.hnsw_ef_search = self.resolve_hnsw_ef_search(ef_search, self.index_meta)
        self._hnsw_cpu_notice_printed = False
        """加载 Chinese-CLIP 预处理器。"""
        self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
        """加载 Chinese-CLIP 模型，并切换到推理模式。"""
        self.model = ChineseCLIPModel.from_pretrained(self.model_name).eval().to(self.device)
        """加载文本向量索引；有显卡时优先尝试搬到显卡。"""
        self.text_index = self.load_index(self.embedding_dir / "text.index", use_gpu_faiss)
        """加载图片向量索引；有显卡时优先尝试搬到显卡。"""
        self.image_index = self.load_index(self.embedding_dir / "image.index", use_gpu_faiss)
        """模型、两个索引和 item_meta 全部加载后执行一次启动交叉校验。"""
        self.validate_startup_compatibility()

    @staticmethod
    def choose_device(requested: str) -> torch.device:
        """
        选择模型推理设备。

        auto 模式优先使用 CUDA；当前无卡环境中会自动回退到处理器，保证代码能做 smoke test。
        """
        """如果用户选择自动模式，就根据 CUDA 是否可用决定。"""
        if requested == "auto":
            """CUDA 可用时返回显卡设备，否则返回处理器设备。"""
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        """用户显式指定设备时直接使用该设备。"""
        return torch.device(requested)

    @staticmethod
    def load_meta(path: Path) -> dict:
        """
        读取检索元信息文件。

        这个文件保存 row_id 到原始文本、图片路径、数据来源等字段的映射。
        """
        """读取结构化元信息。"""
        meta = load_json(path)
        """元信息缺失时直接报错，因为 FAISS 只能返回行号，无法独立展示结果。"""
        if not meta or "items" not in meta:
            """抛出清晰错误，提示用户先完成向量抽取。"""
            raise FileNotFoundError(f"缺少可用元信息文件：{path}")
        """返回元信息字典。"""
        return meta

    @staticmethod
    def load_retrieval_config(model_name: str | Path) -> dict[str, Any]:
        """读取最终模型随附的检索配置；老 baseline 没有该文件时返回空字典。"""

        path = Path(model_name) / "retrieval_config.json"
        if not path.is_file():
            return {}
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"检索配置根节点必须是对象：{path}")
        return payload

    @staticmethod
    def resolve_max_length(requested: int | None, retrieval_config: Mapping[str, Any]) -> int:
        """自动采用模型配置的文本长度，并拒绝调用方传入冲突值。"""

        configured = retrieval_config.get("max_length")
        if configured is not None:
            configured = int(configured)
            if configured <= 0:
                raise ValueError("retrieval_config.max_length 必须是正整数")
            if requested is not None and int(requested) != configured:
                raise MetadataMismatchError(
                    f"查询 max_length={requested} 与最终模型配置 {configured} 不一致"
                )
            return configured
        value = 256 if requested is None else int(requested)
        if value <= 0:
            raise ValueError("max_length 必须是正整数")
        return value

    @staticmethod
    def resolve_hnsw_ef_search(requested: int | None, index_meta: Any) -> int:
        """读取 HNSW efSearch，允许 CLI/环境变量覆盖落盘默认值。"""

        configured = None
        if isinstance(index_meta, Mapping):
            hnsw = index_meta.get("hnsw")
            if isinstance(hnsw, Mapping):
                configured = hnsw.get("ef_search")
        value = requested if requested is not None else configured
        if value is None:
            raise MetadataMismatchError("index_meta.hnsw.ef_search 缺失")
        value = int(value)
        if value <= 0:
            raise ValueError("ef_search 必须是正整数")
        return value

    @staticmethod
    def detect_metadata_mode(item_meta: Mapping[str, Any], index_meta: Any) -> str:
        """识别严格新格式或兼容老 baseline，并拒绝半升级状态。"""

        if index_meta is not None and not isinstance(index_meta, Mapping):
            raise ValueError("index_meta.json 根节点必须是对象")
        required = set(REQUIRED_COMPATIBILITY_FIELDS)
        item_fields = required.intersection(item_meta)
        index_fields = required.intersection(index_meta or {})
        if not item_fields and not index_fields:
            warnings.warn(
                "检测到老 baseline：缺少模型/数据兼容字段，将保持可用但无法验证产物身份",
                RuntimeWarning,
                stacklevel=2,
            )
            return "legacy"
        if item_fields != required or index_fields != required:
            missing_item = sorted(required - item_fields)
            missing_index = sorted(required - index_fields)
            raise MetadataMismatchError(
                "新格式元数据不完整，禁止启动："
                f"item_meta 缺少 {missing_item}；index_meta 缺少 {missing_index}"
            )
        assert_metadata_compatible(index_meta, item_meta, context="索引与 item_meta")
        return "strict"

    def model_dimension(self) -> int:
        """从已加载的最终模型配置读取真实投影维度。"""

        dim = getattr(self.model.config, "projection_dim", None)
        if dim is None:
            dim = getattr(getattr(self.model.config, "text_config", None), "projection_dim", None)
        if dim is None:
            raise MetadataMismatchError("最终模型配置缺少 projection_dim，无法验证索引维度")
        return int(dim)

    def _validate_one_index(self, name: str, index: Any, expected_count: int, expected_dim: int) -> None:
        """校验实际 FAISS 索引规模以及 index_meta 中的单路记录。"""

        actual_count = int(getattr(index, "ntotal", -1))
        actual_dim = int(getattr(index, "d", -1))
        if actual_count != expected_count or actual_dim != expected_dim:
            raise MetadataMismatchError(
                f"{name} 索引规模不一致：count/dim={actual_count}/{actual_dim}，"
                f"期望 {expected_count}/{expected_dim}"
            )
        if not hasattr(index, "hnsw"):
            raise MetadataMismatchError(f"{name} 索引不是 IndexHNSWFlat")
        if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
            raise MetadataMismatchError(f"{name} HNSW 索引必须使用 inner product")
        if int(index.hnsw.efSearch) != self.hnsw_ef_search:
            raise MetadataMismatchError(f"{name} HNSW efSearch 未正确应用")
        if isinstance(self.index_meta, Mapping):
            info = self.index_meta.get(name)
            if isinstance(info, Mapping):
                recorded_count = int(info.get("count", actual_count))
                recorded_dim = int(info.get("dimension", info.get("dim", actual_dim)))
                if recorded_count != actual_count or recorded_dim != actual_dim:
                    raise MetadataMismatchError(f"index_meta.{name} 与实际索引规模不一致")
                recorded_norm = info.get("normalization")
                if self.metadata_mode == "strict" and str(recorded_norm).lower() != "l2":
                    raise MetadataMismatchError(f"index_meta.{name}.normalization 必须为 l2")
            hnsw = self.index_meta.get("hnsw")
            if not isinstance(hnsw, Mapping):
                raise MetadataMismatchError("index_meta.hnsw 缺失")
            expected_m = int(hnsw.get("m", 0))
            expected_ef_construction = int(hnsw.get("ef_construction", 0))
            if expected_m <= 0 or expected_ef_construction <= 0:
                raise MetadataMismatchError("index_meta.hnsw 参数无效")
            if int(index.hnsw.nb_neighbors(0)) != expected_m * 2:
                raise MetadataMismatchError(f"{name} HNSW M 与 index_meta 不一致")
            if int(index.hnsw.efConstruction) != expected_ef_construction:
                raise MetadataMismatchError(f"{name} HNSW efConstruction 与 index_meta 不一致")

    def validate_startup_compatibility(self) -> None:
        """交叉校验最终模型、index_meta、item_meta 与真实索引。"""

        expected_count = len(self.items)
        if int(self.meta.get("count", expected_count)) != expected_count:
            raise MetadataMismatchError("item_meta.count 与 items 列表长度不一致")
        model_dim = self.model_dimension()

        if self.metadata_mode == "strict":
            if self.index_meta.get("index_type") != "IndexHNSWFlat":
                raise MetadataMismatchError("index_meta.index_type 必须为 IndexHNSWFlat")
            if self.index_meta.get("metric") != "inner_product":
                raise MetadataMismatchError("index_meta.metric 必须为 inner_product")
            model_dir = Path(self.model_name)
            if not model_dir.is_dir():
                raise MetadataMismatchError("新格式索引必须使用可计算内容指纹的本地最终模型目录")
            normalization = str(self.retrieval_config.get("normalization", "l2")).lower()
            if normalization != "l2":
                raise MetadataMismatchError(f"查询流水线只支持 l2 归一化，模型配置为 {normalization!r}")
            configured_dim = self.retrieval_config.get("dimension")
            if configured_dim is not None and int(configured_dim) != model_dim:
                raise MetadataMismatchError("retrieval_config.dimension 与实际模型维度不一致")
            expected = dict(self.meta)
            expected.update(
                {
                    "model_fingerprint": compute_model_fingerprint(model_dir),
                    "dimension": model_dim,
                    "normalization": normalization,
                }
            )
            assert_metadata_compatible(self.meta, expected, context="最终模型与 item_meta")
            assert_metadata_compatible(self.index_meta, expected, context="最终模型与索引")
            for label, payload in (("item_meta", self.meta), ("index_meta", self.index_meta)):
                if payload.get("max_length") is not None and int(payload["max_length"]) != self.max_length:
                    raise MetadataMismatchError(f"{label}.max_length 与最终模型 retrieval_config 不一致")
            if int(self.index_meta.get("count", expected_count)) != expected_count:
                raise MetadataMismatchError("index_meta.count 与 item_meta 不一致")
        else:
            """老 baseline 仍检查不会影响兼容性的基本维度，避免 FAISS 在查询时才崩溃。"""
            expected_dim = int(getattr(self.text_index, "d", model_dim))
            if model_dim != expected_dim:
                raise MetadataMismatchError("老 baseline 的模型维度与索引维度不一致")

        expected_dim = int(self.meta.get("dimension", model_dim))
        self._validate_one_index("text", self.text_index, expected_count, expected_dim)
        self._validate_one_index("image", self.image_index, expected_count, expected_dim)

    def load_index(self, path: Path, use_gpu: bool):
        """
        读取 FAISS 索引，并在可用时尝试搬到显卡。

        当前代码默认面向后续有卡运行；如果无卡或 GPU FAISS 不可用，会自动保留处理器侧索引。
        """
        """先读取落盘的处理器侧索引。"""
        index = faiss.read_index(str(path))
        """FAISS HNSW 是 CPU 图索引，保留在 CPU 并应用查询深度。"""
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = self.hnsw_ef_search
            if use_gpu and not self._hnsw_cpu_notice_printed:
                print("检测到 IndexHNSWFlat：索引保持在 CPU，Chinese-CLIP 编码仍使用所选设备。", flush=True)
                self._hnsw_cpu_notice_printed = True
            return index
        """如果调用方关闭显卡 FAISS，就直接返回处理器索引。"""
        if not use_gpu:
            """返回处理器侧索引。"""
            return index
        """尝试判断 FAISS 能看到几张显卡；异常时按无显卡处理。"""
        try:
            """读取 FAISS 可用显卡数量。"""
            gpu_count = faiss.get_num_gpus()
        except Exception:
            """无法读取显卡数量时，认为 GPU FAISS 不可用。"""
            gpu_count = 0
        """无显卡时直接使用处理器索引。"""
        if gpu_count <= 0:
            """返回处理器侧索引，保证无卡环境不崩。"""
            return index
        """有显卡时尝试把索引复制到第零张显卡。"""
        try:
            """创建显卡索引依赖的资源对象。"""
            resources = faiss.StandardGpuResources()
            """持有资源对象，避免函数返回后被回收。"""
            self.gpu_resources.append(resources)
            """把处理器索引复制到显卡。"""
            return faiss.index_cpu_to_gpu(resources, 0, index)
        except Exception as exc:
            """GPU 转换失败时打印提示，并自动回退处理器索引。"""
            print(f"FAISS 显卡索引不可用，已回退处理器索引：{exc}", flush=True)
            """返回处理器侧索引。"""
            return index

    @staticmethod
    def unwrap_features(features):
        """
        兼容不同 transformers 版本的返回格式。

        有的版本直接返回张量，有的版本返回对象或元组；这里统一抽出真正的向量张量。
        """
        """如果返回值已经是张量，就直接返回。"""
        if isinstance(features, torch.Tensor):
            """返回张量。"""
            return features
        """按常见字段名查找向量张量。"""
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            """读取当前字段。"""
            value = getattr(features, attr, None)
            """如果字段值是张量，就认为它是目标向量。"""
            if isinstance(value, torch.Tensor):
                """返回找到的张量。"""
                return value
        """兼容元组或列表返回格式。"""
        if isinstance(features, (tuple, list)) and features and isinstance(features[0], torch.Tensor):
            """返回序列中的第一个张量。"""
            return features[0]
        """所有兼容路径都失败时抛错。"""
        raise TypeError(f"无法从模型输出中找到向量张量，输出类型为 {type(features)!r}")

    def move_to_device(self, encoded: dict) -> dict:
        """
        把预处理后的张量输入移动到模型所在设备。
        """
        """逐项移动张量；非张量值保持原样。"""
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in encoded.items()}

    def finish_features(self, features) -> np.ndarray:
        """
        把模型输出整理成 FAISS 可检索的二维 float32 矩阵。
        """
        """抽出真正的向量张量。"""
        features = self.unwrap_features(features)
        """做二范数归一化，使内积分数等价于余弦相似度。"""
        features = F.normalize(features, p=2, dim=-1)
        """转回处理器内存，并转换成 FAISS 需要的 float32。"""
        return features.detach().float().cpu().numpy().astype("float32")

    def encode_text(self, query: str) -> np.ndarray:
        """
        把文本查询编码成 Chinese-CLIP 文本向量。
        """
        """把文本转成模型输入张量。"""
        encoded = self.processor(text=[query], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        """移动到模型所在设备。"""
        encoded = self.move_to_device(encoded)
        """关闭梯度计算，只做推理。"""
        with torch.inference_mode():
            """显卡环境下按需使用半精度推理。"""
            if self.device.type == "cuda" and self.fp16:
                """在 CUDA 自动混合精度上下文中编码文本。"""
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    """调用文本编码接口得到查询向量。"""
                    features = self.model.get_text_features(**encoded)
            else:
                """处理器或关闭半精度时使用默认精度编码文本。"""
                features = self.model.get_text_features(**encoded)
        """整理并返回查询向量。"""
        return self.finish_features(features)

    def open_image(self, image_path: str | Path) -> Image.Image:
        """
        打开查询图片并转成三通道格式。

        相对路径会按项目根目录解析；绝对路径通常来自上传文件，可以直接读取。
        """
        """把图片路径解析为真实文件路径。"""
        path = resolve_path(self.project_root, image_path)
        """打开图片并统一为 RGB 三通道。"""
        return Image.open(path).convert("RGB")

    def encode_image(self, image_path: str | Path) -> np.ndarray:
        """
        把图片查询编码成 Chinese-CLIP 图片向量。
        """
        """读取查询图片。"""
        image = self.open_image(image_path)
        """把图片转成模型输入张量。"""
        encoded = self.processor(images=[image], return_tensors="pt")
        """移动到模型所在设备。"""
        encoded = self.move_to_device(encoded)
        """关闭梯度计算，只做推理。"""
        with torch.inference_mode():
            """显卡环境下按需使用半精度推理。"""
            if self.device.type == "cuda" and self.fp16:
                """在 CUDA 自动混合精度上下文中编码图片。"""
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    """调用图片编码接口得到查询向量。"""
                    features = self.model.get_image_features(**encoded)
            else:
                """处理器或关闭半精度时使用默认精度编码图片。"""
                features = self.model.get_image_features(**encoded)
        """整理并返回查询向量。"""
        return self.finish_features(features)

    @staticmethod
    def collect_scores(distances, indices) -> Dict[int, float]:
        """
        把 FAISS 返回的距离和行号整理成 row_id 到分数的字典。
        """
        """准备分数字典。"""
        scores: Dict[int, float] = {}
        """遍历当前 query 的所有召回结果。"""
        for score, row_id in zip(distances[0].tolist(), indices[0].tolist()):
            """FAISS 用负行号表示无效结果，需要跳过。"""
            if row_id < 0:
                """跳过无效行号。"""
                continue
            """同一行如果重复出现，保留更高分数。"""
            scores[int(row_id)] = max(float(score), scores.get(int(row_id), float("-inf")))
        """返回整理后的分数。"""
        return scores

    def search_index(self, index, query_embedding: np.ndarray, top_k: int) -> Dict[int, float]:
        """
        在指定索引里检索一个查询向量。
        """
        """执行 FAISS 检索。"""
        distances, indices = index.search(np.asarray(query_embedding, dtype="float32"), top_k)
        """整理并返回 row_id 到分数的映射。"""
        return self.collect_scores(distances, indices)

    @staticmethod
    def clean_text(query: Optional[str]) -> Optional[str]:
        """
        清理用户输入文本。
        """
        """空值直接返回空。"""
        if query is None:
            """表示没有文本查询。"""
            return None
        """去掉首尾空白。"""
        query = str(query).strip()
        """空字符串按没有文本查询处理。"""
        return query or None

    @staticmethod
    def clean_image_path(image_path: Optional[str | Path]) -> Optional[str]:
        """
        清理用户输入图片路径。
        """
        """空值直接返回空。"""
        if image_path is None:
            """表示没有图片查询。"""
            return None
        """去掉首尾空白。"""
        value = str(image_path).strip()
        """空字符串按没有图片查询处理。"""
        return value or None

    def fuse_route_scores(
        self,
        route_scores: Mapping[str, Mapping[int, float]],
        has_text: bool,
        has_image: bool,
        alpha: float,
    ) -> List[dict]:
        """把四路候选按固定 late fusion 公式合并并排序。"""

        candidates = sorted(set().union(*(scores.keys() for scores in route_scores.values())))
        denominator = max(int(has_text) + int(has_image), 1)
        results: List[dict] = []
        for row_id in candidates:
            if row_id < 0 or row_id >= len(self.items):
                raise MetadataMismatchError(f"索引返回越界 row_id={row_id}，item_meta 与索引未对齐")
            score_tt = route_scores["score_tt"].get(row_id, 0.0)
            score_ti = route_scores["score_ti"].get(row_id, 0.0)
            score_it = route_scores["score_it"].get(row_id, 0.0)
            score_ii = route_scores["score_ii"].get(row_id, 0.0)
            score_text_index = ((score_tt if has_text else 0.0) + (score_it if has_image else 0.0)) / denominator
            score_image_index = ((score_ti if has_text else 0.0) + (score_ii if has_image else 0.0)) / denominator
            score_mm = alpha * score_text_index + (1.0 - alpha) * score_image_index
            item = dict(self.items[row_id])
            item.update(
                {
                    "row_id": row_id,
                    "score_tt": score_tt,
                    "score_ti": score_ti,
                    "score_it": score_it,
                    "score_ii": score_ii,
                    "score_text_index": score_text_index,
                    "score_image_index": score_image_index,
                    "score_mm": score_mm,
                }
            )
            results.append(item)
        results.sort(key=lambda row: row["score_mm"], reverse=True)
        return results

    def search(
        self,
        query: Optional[str] = None,
        image_path: Optional[str | Path] = None,
        alpha: float = 0.5,
        top_k_recall: int = 100,
        top_k_final: int = 10,
    ) -> List[dict]:
        """
        执行文本、图片或图文联合检索。

        文本和图片至少需要提供一个；如果都提供，就形成四路召回分数。
        """
        """清理文本查询。"""
        query = self.clean_text(query)
        """清理图片查询路径。"""
        image_path = self.clean_image_path(image_path)
        """如果两种查询都缺失，就抛出清晰错误。"""
        if not query and not image_path:
            """提示调用方至少提供一种查询输入。"""
            raise ValueError("请至少提供 query 或 image_path 中的一个")
        if top_k_recall <= 0:
            raise ValueError("top_k_recall 必须是正整数")
        if top_k_final <= 0:
            return []
        """记录文本查询是否可用。"""
        has_text = query is not None
        """记录图片查询是否可用。"""
        has_image = image_path is not None
        text_embedding = None
        image_embedding = None
        """查询只编码一次；固定召回 top_k_recall 个图文对，不做自适应扩充。"""
        if has_text:
            text_embedding = self.encode_text(query)
        if has_image:
            image_embedding = self.encode_image(image_path)

        recall_k = max(int(top_k_recall), int(top_k_final))
        route_scores: Dict[str, Dict[int, float]] = {
            "score_tt": {},
            "score_ti": {},
            "score_it": {},
            "score_ii": {},
        }
        if has_text:
            route_scores["score_tt"] = self.search_index(self.text_index, text_embedding, recall_k)
            route_scores["score_ti"] = self.search_index(self.image_index, text_embedding, recall_k)
        if has_image:
            route_scores["score_it"] = self.search_index(self.text_index, image_embedding, recall_k)
            route_scores["score_ii"] = self.search_index(self.image_index, image_embedding, recall_k)
        """四路分数融合并按 score_mm 降序排列。"""
        ranked = self.fuse_route_scores(route_scores, has_text, has_image, alpha)
        """取前 top_k_recall 个图文对作为候选池。"""
        candidates = ranked[:recall_k]
        """MUGE 一文多图聚合（Flickr/COCO 保持独立），聚合后重新排序。"""
        aggregated = aggregate_results(candidates, self.items, self.row_to_text_id)
        """返回前 top_k_final 个文档级结果。"""
        return aggregated[:top_k_final]

    @staticmethod
    def dumps(results: List[dict]) -> str:
        """
        把搜索结果转成中文友好的结构化文本。
        """
        """序列化搜索结果，保留中文原文。"""
        return json.dumps(results, ensure_ascii=False, indent=2)
