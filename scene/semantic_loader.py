"""
Semantic Data Loader for Phase 3: Loading Phase 2 Outputs.

Loads pre-computed semantic label maps and CLIP feature maps from the
Phase 2 extraction pipeline, making them available as supervision
signals during 4D-GS training.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class SemanticDataManager:
    """Manages loading and caching of Phase 2 semantic supervision data.

    Loads label maps (PNG), CLIP feature maps (NPY), and metadata from
    the semantics/ directory produced by Phase 2 extraction.

    Attributes:
        semantics_dir: Path to the semantics/ directory.
        metadata: Pipeline metadata from metadata.json.
        num_classes: Number of semantic classes (from text_prompts).
        feature_dim: Dimensionality of CLIP features (e.g., 512).
        class_names: Mapping from label ID to class name string.
        text_features: Precomputed CLIP text embeddings (K, D).
    """

    def __init__(self, dataset_path: str, device: str = "cuda"):
        """Initialize the semantic data manager.

        Args:
            dataset_path: Root D-NeRF dataset directory. Expects a
                semantics/ subdirectory with Phase 2 outputs.
            device: PyTorch device for loaded tensors.
        """
        self.dataset_path = Path(dataset_path)
        self.semantics_dir = self.dataset_path / "semantics"
        self.device = device
        self.enabled = False

        # Caches
        self._label_cache: Dict[str, torch.Tensor] = {}
        self._feature_cache: Dict[str, torch.Tensor] = {}

        # Try to load metadata
        self.metadata = None
        self.num_classes = 0
        self.feature_dim = 0
        self.class_names = {}
        self.text_features = None

        if self.semantics_dir.exists():
            self._load_metadata()

    def _load_metadata(self):
        """Load metadata.json and text features from semantics/ directory."""
        metadata_path = self.semantics_dir / "metadata.json"
        if not metadata_path.exists():
            print(f"[Semantic] No metadata.json found in {self.semantics_dir}")
            return

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        config = self.metadata.get("config", {})
        self.num_classes = len(config.get("text_prompts", []))
        self.feature_dim = config.get("clip_feature_dim", 512)
        self.class_names = self.metadata.get("class_mapping", {})

        # Load text features if available
        text_feat_path = self.semantics_dir / "clip_text_features.npy"
        if text_feat_path.exists():
            self.text_features = torch.from_numpy(
                np.load(str(text_feat_path))
            ).float().to(self.device)

        # Check if label/feature files exist
        labels_dir = self.semantics_dir / "labels"
        features_dir = self.semantics_dir / "features"

        has_labels = labels_dir.exists() and any(labels_dir.glob("*.png"))
        has_features = features_dir.exists() and any(features_dir.glob("*.npy"))

        if has_labels or has_features:
            self.enabled = True
            stats = self.metadata.get("statistics", {})
            print(f"[Semantic] Loaded Phase 2 data:")
            print(f"  Classes: {self.num_classes} ({list(self.class_names.values())})")
            print(f"  Feature dim: {self.feature_dim}")
            print(f"  Frames: {stats.get('total_frames_processed', '?')}")
        else:
            print(f"[Semantic] No label/feature files found in {self.semantics_dir}")

    def get_label_map(self, frame_name: str) -> Optional[torch.Tensor]:
        """Load a per-pixel label map for a given frame.

        Args:
            frame_name: Frame identifier (e.g., "r_0"). The loader
                strips directory prefixes and extensions automatically.

        Returns:
            (H, W) long tensor of class labels, or None if not found.
        """
        if not self.enabled:
            return None

        # Normalize frame name (strip path prefix like "./train/")
        frame_name = Path(frame_name).stem

        if frame_name in self._label_cache:
            return self._label_cache[frame_name]

        label_path = self.semantics_dir / "labels" / f"{frame_name}_labels.png"
        if not label_path.exists():
            return None

        label_map = np.array(Image.open(str(label_path)))
        label_tensor = torch.from_numpy(label_map).long().to(self.device)

        self._label_cache[frame_name] = label_tensor
        return label_tensor

    def get_feature_map(self, frame_name: str) -> Optional[torch.Tensor]:
        """Load a per-pixel CLIP feature map for a given frame.

        Args:
            frame_name: Frame identifier (e.g., "r_0").

        Returns:
            (D, H, W) float tensor of CLIP features, or None if not found.
            Note: returned in (D, H, W) format for consistency with
            rendered feature maps.
        """
        if not self.enabled:
            return None

        frame_name = Path(frame_name).stem

        if frame_name in self._feature_cache:
            return self._feature_cache[frame_name]

        feature_path = self.semantics_dir / "features" / f"{frame_name}_features.npy"
        if not feature_path.exists():
            return None

        feature_map = np.load(str(feature_path)).astype(np.float32)
        # feature_map is (H, W, D) -> permute to (D, H, W)
        feature_tensor = torch.from_numpy(feature_map).permute(2, 0, 1).to(self.device)

        self._feature_cache[frame_name] = feature_tensor
        return feature_tensor

    def get_semantic_supervision(
        self, frame_name: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get both label map and feature map for a frame.

        Args:
            frame_name: Frame identifier.

        Returns:
            Tuple of (label_map, feature_map), either may be None.
        """
        return self.get_label_map(frame_name), self.get_feature_map(frame_name)

    def clear_cache(self):
        """Clear cached label and feature maps to free GPU memory."""
        self._label_cache.clear()
        self._feature_cache.clear()
        torch.cuda.empty_cache()
