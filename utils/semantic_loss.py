"""
Semantic Loss Functions for Phase 3: Semantic Lifting into 4D Gaussians.

Provides loss functions to supervise per-Gaussian semantic features using
the 2D semantic labels and CLIP features extracted in Phase 2.

Supports two modes:
  - Cross-entropy loss against integer label maps
  - Cosine similarity loss against dense CLIP feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def semantic_cross_entropy_loss(
    rendered_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Cross-entropy loss between rendered semantic logits and GT label map.

    Args:
        rendered_logits: (K, H, W) rendered per-pixel class logits from
            semantic Gaussian rasterization. K = number of semantic classes.
        gt_labels: (H, W) integer label map from Phase 2 (uint8).
            Label 0 = unlabeled/background (ignored by default).
            Labels 1..K correspond to semantic classes.
        ignore_index: Label value to ignore in loss computation.

    Returns:
        Scalar cross-entropy loss.
    """
    # rendered_logits: (K, H, W) -> (1, K, H, W) for F.cross_entropy
    logits = rendered_logits.unsqueeze(0)  # (1, K, H, W)
    # gt_labels: (H, W) -> (1, H, W) as long tensor
    target = gt_labels.unsqueeze(0).long()  # (1, H, W)

    loss = F.cross_entropy(
        logits, target,
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss


def semantic_cosine_loss(
    rendered_features: torch.Tensor,
    gt_features: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Cosine similarity loss between rendered and GT feature maps.

    Maximizes cosine similarity between rendered per-pixel features
    and the ground-truth CLIP features from Phase 2.

    Args:
        rendered_features: (D, H, W) rendered per-pixel semantic features.
        gt_features: (D, H, W) ground-truth CLIP feature map.
        valid_mask: Optional (H, W) boolean mask. Only compute loss where True.
            Useful for ignoring unlabeled/background pixels.

    Returns:
        Scalar cosine similarity loss (1 - mean_similarity).
    """
    D, H, W = rendered_features.shape

    # Reshape to (H*W, D)
    pred = rendered_features.permute(1, 2, 0).reshape(-1, D)  # (H*W, D)
    target = gt_features.permute(1, 2, 0).reshape(-1, D)      # (H*W, D)

    if valid_mask is not None:
        mask_flat = valid_mask.reshape(-1)  # (H*W,)
        pred = pred[mask_flat]
        target = target[mask_flat]

    if pred.shape[0] == 0:
        return torch.tensor(0.0, device=rendered_features.device)

    # Normalize
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity: 1 = identical, -1 = opposite
    similarity = (pred * target).sum(dim=-1)  # (N,)

    # Loss = 1 - mean_similarity (minimizing pushes toward similarity = 1)
    loss = 1.0 - similarity.mean()
    return loss


def semantic_mse_loss(
    rendered_features: torch.Tensor,
    gt_features: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """MSE loss between rendered and GT feature maps.

    Args:
        rendered_features: (D, H, W) rendered per-pixel semantic features.
        gt_features: (D, H, W) ground-truth feature map.
        valid_mask: Optional (H, W) boolean mask.

    Returns:
        Scalar MSE loss.
    """
    if valid_mask is not None:
        # Expand mask to match feature dims
        mask = valid_mask.unsqueeze(0).expand_as(rendered_features)  # (D, H, W)
        diff = (rendered_features - gt_features) ** 2
        loss = diff[mask].mean()
    else:
        loss = F.mse_loss(rendered_features, gt_features)

    return loss


def compute_semantic_loss(
    rendered_semantics: torch.Tensor,
    gt_labels: torch.Tensor = None,
    gt_features: torch.Tensor = None,
    loss_type: str = "cross_entropy",
    ignore_index: int = 0,
) -> torch.Tensor:
    """Unified semantic loss computation.

    Args:
        rendered_semantics: Rendered semantic output from Gaussian rasterization.
            Shape depends on loss_type:
            - "cross_entropy": (K, H, W) class logits
            - "cosine": (D, H, W) feature vectors
        gt_labels: (H, W) integer label map (required for cross_entropy).
        gt_features: (D, H, W) CLIP feature map (required for cosine/mse).
        loss_type: One of "cross_entropy", "cosine", "mse".
        ignore_index: Label to ignore in cross-entropy.

    Returns:
        Scalar semantic loss value.
    """
    if loss_type == "cross_entropy":
        assert gt_labels is not None, "gt_labels required for cross_entropy loss"
        return semantic_cross_entropy_loss(rendered_semantics, gt_labels, ignore_index)
    elif loss_type == "cosine":
        assert gt_features is not None, "gt_features required for cosine loss"
        valid_mask = None
        if gt_labels is not None:
            valid_mask = gt_labels > 0  # ignore unlabeled
        return semantic_cosine_loss(rendered_semantics, gt_features, valid_mask)
    elif loss_type == "mse":
        assert gt_features is not None, "gt_features required for mse loss"
        valid_mask = None
        if gt_labels is not None:
            valid_mask = gt_labels > 0
        return semantic_mse_loss(rendered_semantics, gt_features, valid_mask)
    else:
        raise ValueError(f"Unknown semantic loss type: {loss_type}")
