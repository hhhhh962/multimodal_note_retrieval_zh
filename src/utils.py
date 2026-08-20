"""
项目一通用工具函数。

这里集中放多个脚本都会用到的小功能，例如目录创建、结构化文件读写、路径解析、
文本字段整理和向量归一化。业务脚本只需要调用这些函数，不必在每个文件里重复实现。
"""

"""引入结构化文本工具，用来读取和写入结构化文件。"""
import json
"""引入计时工具，用来统计任务运行耗时。"""
import time
"""引入路径对象，方便跨平台拼接和判断路径。"""
from pathlib import Path
"""引入类型标注，帮助读者理解函数输入和输出。"""
from typing import Any, Dict, Iterator, List, Sequence

"""引入矩阵计算库，用来处理向量归一化。"""
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """
    确保目录存在。

    如果上级目录不存在，会一起创建；返回路径对象，方便调用方继续拼接子路径。
    """
    """把字符串路径或路径对象统一转换成路径对象。"""
    target = Path(path)
    """创建目标目录；目录已存在时不报错。"""
    target.mkdir(parents=True, exist_ok=True)
    """返回标准化后的路径对象。"""
    return target


def read_jsonl(path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    一次性读取“一行一条记录”的结构化文件。

    limit 用于小规模试跑，例如只读取前一百条，避免每次调试都跑完整数据。
    """
    """准备列表，用来保存读取到的每条记录。"""
    rows: List[Dict[str, Any]] = []
    """用只读模式打开输入文件，并使用中文友好的编码。"""
    with Path(path).open("r", encoding="utf-8") as f:
        """逐行读取文件内容。"""
        for line in f:
            """跳过空行，只处理有内容的记录。"""
            if line.strip():
                """把当前行从文本解析成字典，并加入结果列表。"""
                rows.append(json.loads(line))
                """如果设置了读取上限，并且已经达到上限，就提前停止。"""
                if limit is not None and len(rows) >= limit:
                    """跳出循环，避免继续读取后续大文件内容。"""
                    break
    """返回读取到的记录列表。"""
    return rows


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """
    流式读取“一行一条记录”的结构化文件。

    当数据继续变大时，可以边读边处理，减少一次性内存占用。
    """
    """用只读模式打开输入文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        """逐行读取文件。"""
        for line in f:
            """跳过空行，只产出有效记录。"""
            if line.strip():
                """把当前行解析成字典，并作为生成器结果返回给调用方。"""
                yield json.loads(line)


def save_json(obj: Any, path: str | Path) -> None:
    """
    原子化保存结构化文本。

    先写到临时文件，再替换目标文件，避免进程中断时留下半个状态文件。
    """
    """把输出路径统一转换成路径对象。"""
    path = Path(path)
    """确保输出文件所在目录存在。"""
    ensure_dir(path.parent)
    """构造临时文件路径，先把内容写到这里。"""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    """打开临时文件，准备写入结构化文本。"""
    with tmp_path.open("w", encoding="utf-8") as f:
        """写入结构化数据；中文保持原样。"""
        json.dump(obj, f, ensure_ascii=False)
    """用临时文件替换目标文件，降低中断时写坏文件的风险。"""
    tmp_path.replace(path)


def load_json(path: str | Path, default: Any = None) -> Any:
    """
    读取结构化文本。

    如果文件不存在，返回调用方给出的默认值；适合读取可选的断点状态文件。
    """
    """把输入路径统一转换成路径对象。"""
    path = Path(path)
    """如果文件不存在，就直接返回默认值。"""
    if not path.exists():
        """返回调用方指定的默认结果。"""
        return default
    """打开存在的结构化文本文件。"""
    with path.open("r", encoding="utf-8") as f:
        """解析文件内容并返回。"""
        return json.load(f)


def batched(seq: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """
    把一个序列切成多个小批次。

    这个工具用于需要分批处理的场景，可以避免一次提交过多数据给模型或索引库。
    """
    """从序列开头开始，每次移动一个批次长度。"""
    for start in range(0, len(seq), batch_size):
        """返回当前批次对应的切片。"""
        yield seq[start : start + batch_size]


def resolve_path(project_root: str | Path, maybe_relative: str | Path) -> Path:
    """
    把数据里的图片路径解析成真实文件路径。

    如果传入的是绝对路径，直接返回；如果是相对路径，则拼到项目根目录下面。
    """
    """把可能是字符串的路径统一转换成路径对象。"""
    path = Path(maybe_relative)
    """如果已经是绝对路径，就不再拼接项目根目录。"""
    if path.is_absolute():
        """直接返回原路径。"""
        return path
    """把相对路径拼接到项目根目录下面。"""
    return Path(project_root) / path


def item_text(item: Dict[str, Any]) -> str:
    """
    统一取得一条样本的检索文本。

    优先使用主文本字段；如果不存在，就把标题和正文拼起来。若仍然为空，返回一个空格，
    避免分词器收到空字符串后行为不稳定。
    """
    """优先读取样本里的主文本字段。"""
    text = item.get("text")
    """如果主文本字段存在且非空，就直接返回它。"""
    if text:
        """确保返回值是字符串。"""
        return str(text)
    """主文本不存在时，尝试读取标题和正文。"""
    parts = [str(item.get("title", "")).strip(), str(item.get("content", "")).strip()]
    """把非空的标题和正文拼成一个检索文本。"""
    text = " ".join(part for part in parts if part)
    """如果拼接后仍为空，就返回一个空格作为兜底文本。"""
    return text or " "


def l2_normalize_np(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对矩阵逐行做二范数归一化。

    向量长度归一后，内积检索就等价于余弦相似度检索。
    """
    """计算每一行向量的二范数。"""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    """用范数对原向量逐行相除；下限值用于避免除零。"""
    return values / np.maximum(norms, eps)


class Timer:
    """
    简单计时器。

    用于进度条和日志里显示任务已经运行了多久。
    """

    def __init__(self) -> None:
        """记录计时器创建时的时间戳。"""
        self.start = time.time()

    def elapsed(self) -> float:
        """
        返回从创建计时器开始到现在经过的秒数。
        """
        """当前时间减去开始时间，得到已运行秒数。"""
        return time.time() - self.start

    def elapsed_text(self) -> str:
        """
        把秒数格式化为时、分、秒，方便阅读日志。
        """
        """把浮点秒数转成整数秒。"""
        seconds = int(self.elapsed())
        """把总秒数拆成小时和剩余秒数。"""
        hours, rem = divmod(seconds, 3600)
        """把剩余秒数继续拆成分钟和秒。"""
        minutes, seconds = divmod(rem, 60)
        """按两位数格式返回时分秒文本。"""
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
