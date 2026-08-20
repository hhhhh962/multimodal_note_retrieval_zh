"""Chinese-CLIP 微调使用的多正样本双向对比损失。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


def multi_positive_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    dim: int = -1,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """为每个锚点计算 ``-log(sum(exp(positive)) / sum(exp(all)))``。

    与仅承认对角线的交叉熵不同，``positive_mask`` 中所有 ``True`` 位置都视为
    正确匹配。log-sum-exp 使用 float32 计算，以提高 BF16/FP16 训练的数值
    稳定性，同时保留到 ``logits`` 的梯度。
    """

    if logits.ndim != 2:
        raise ValueError(f"logits must be rank 2, got shape {tuple(logits.shape)}")
    if positive_mask.shape != logits.shape:
        raise ValueError(
            "positive_mask must have the same shape as logits: "
            f"{tuple(positive_mask.shape)} != {tuple(logits.shape)}"
        )
    dim = dim if dim >= 0 else logits.ndim + dim
    if dim not in (0, 1):
        raise ValueError("dim must select the text or image axis of a rank-2 tensor")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"unsupported reduction: {reduction!r}")

    mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    anchors_with_positives = mask.any(dim=dim)
    if not bool(anchors_with_positives.all().item()):
        missing = (~anchors_with_positives).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"anchors without a positive at indices {missing}")

    working_logits = logits.float()
    log_denominator = torch.logsumexp(working_logits, dim=dim)
    positive_logits = working_logits.masked_fill(~mask, -torch.inf)
    log_numerator = torch.logsumexp(positive_logits, dim=dim)
    losses = log_denominator - log_numerator

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def symmetric_multi_positive_loss(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    positive_mask: torch.Tensor,
    logit_scale: torch.Tensor | float,
    *,
    eps: float = 1e-6,
    return_directional: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对双塔特征执行 L2 归一化，并平均文搜图与图搜文损失。

    ``logit_scale`` 是 Hugging Face ``ChineseCLIPModel`` 使用的对数域参数，
    实际相乘的温度系数为 ``exp(logit_scale)``。正样本矩阵应具有
    ``[文本数量, 图片数量]`` 的形状。
    """

    if text_features.ndim != 2 or image_features.ndim != 2:
        raise ValueError("text_features and image_features must both be rank 2")
    if text_features.shape[1] != image_features.shape[1]:
        raise ValueError(
            "text/image embedding dimensions differ: "
            f"{text_features.shape[1]} != {image_features.shape[1]}"
        )
    expected_mask_shape = (text_features.shape[0], image_features.shape[0])
    if tuple(positive_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"positive_mask shape {tuple(positive_mask.shape)} does not match "
            f"{expected_mask_shape}"
        )

    normalized_text = F.normalize(text_features.float(), p=2, dim=-1, eps=eps)
    normalized_image = F.normalize(image_features.float(), p=2, dim=-1, eps=eps)
    if torch.is_tensor(logit_scale):
        if logit_scale.numel() != 1:
            raise ValueError("logit_scale must be a scalar")
        scale = logit_scale.float().reshape(()).exp()
    else:
        scale = normalized_text.new_tensor(float(logit_scale)).exp()

    logits_per_text = scale * normalized_text @ normalized_image.transpose(0, 1)
    mask = positive_mask.to(device=logits_per_text.device, dtype=torch.bool)
    text_to_image = multi_positive_cross_entropy(
        logits_per_text, mask, dim=-1, reduction="mean"
    )
    image_to_text = multi_positive_cross_entropy(
        logits_per_text.transpose(0, 1),
        mask.transpose(0, 1),
        dim=-1,
        reduction="mean",
    )
    loss = 0.5 * (text_to_image + image_to_text)
    if return_directional:
        return loss, text_to_image, image_to_text
    return loss


# 为采用 CLIP 风格命名的训练入口提供明确别名。
multi_positive_clip_loss = symmetric_multi_positive_loss


__all__ = [
    "multi_positive_clip_loss",
    "multi_positive_cross_entropy",
    "symmetric_multi_positive_loss",
]
