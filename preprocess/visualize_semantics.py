"""
Visualization Utilities for Semantic Feature Extraction Results.

Provides functions to overlay masks on images, render color-coded label maps,
visualize CLIP feature similarity heatmaps, and compile results into videos.

Usage:
    python -m preprocess.visualize_semantics \
        --dataset_path data/dnerf/bouncingballs \
        --output_path visualizations/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Matplotlib imports with non-interactive backend fallback
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless/Colab
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Color palette for semantic classes ───────────────────────────────
# 20 distinct colors for up to 20 classes (index 0 = unlabeled/transparent)
SEMANTIC_COLORS = np.array([
    [0,   0,   0  ],   # 0: unlabeled / background (black)
    [255, 0,   0  ],   # 1: class 1 (red)
    [0,   255, 0  ],   # 2: class 2 (green)
    [0,   0,   255],   # 3: class 3 (blue)
    [255, 255, 0  ],   # 4: class 4 (yellow)
    [255, 0,   255],   # 5: class 5 (magenta)
    [0,   255, 255],   # 6: class 6 (cyan)
    [255, 128, 0  ],   # 7: class 7 (orange)
    [128, 0,   255],   # 8: class 8 (purple)
    [0,   128, 255],   # 9: class 9 (sky blue)
    [255, 128, 128],   # 10
    [128, 255, 128],   # 11
    [128, 128, 255],   # 12
    [255, 255, 128],   # 13
    [255, 128, 255],   # 14
    [128, 255, 255],   # 15
    [192, 64,  0  ],   # 16
    [0,   192, 64 ],   # 17
    [64,  0,   192],   # 18
    [192, 192, 0  ],   # 19
], dtype=np.uint8)


def load_metadata(semantics_dir: str) -> Dict:
    """Load metadata.json from the semantics output directory."""
    metadata_path = Path(semantics_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {semantics_dir}")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_label_map(semantics_dir: str, frame_name: str) -> np.ndarray:
    """Load a per-pixel label map for a given frame."""
    path = Path(semantics_dir) / "labels" / f"{frame_name}_labels.png"
    return np.array(Image.open(path))


def load_feature_map(semantics_dir: str, frame_name: str) -> np.ndarray:
    """Load a per-pixel CLIP feature map for a given frame."""
    path = Path(semantics_dir) / "features" / f"{frame_name}_features.npy"
    return np.load(path)


def load_masks(semantics_dir: str, frame_name: str) -> Tuple[List[np.ndarray], Dict]:
    """Load individual masks and their metadata for a given frame."""
    path = Path(semantics_dir) / "masks" / f"{frame_name}_masks.npz"
    data = np.load(path, allow_pickle=True)

    metadata = json.loads(str(data["metadata"]))
    num_masks = metadata["num_masks"]

    masks = []
    for i in range(num_masks):
        key = f"mask_{i}"
        if key in data:
            masks.append(data[key].astype(bool))

    return masks, metadata


# ─────────────────────────────────────────────────────────────────────
# Visualization Functions
# ─────────────────────────────────────────────────────────────────────

def colorize_label_map(
    label_map: np.ndarray,
    class_names: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Convert an integer label map to an RGB color image.

    Args:
        label_map: (H, W) uint8 array of class IDs.
        class_names: Optional mapping from label ID to class name.

    Returns:
        color_map: (H, W, 3) uint8 RGB image.
    """
    H, W = label_map.shape
    color_map = np.zeros((H, W, 3), dtype=np.uint8)

    unique_labels = np.unique(label_map)
    for label_id in unique_labels:
        color_idx = label_id % len(SEMANTIC_COLORS)
        color_map[label_map == label_id] = SEMANTIC_COLORS[color_idx]

    return color_map


def overlay_masks_on_image(
    image: np.ndarray,
    label_map: np.ndarray,
    alpha: float = 0.45,
    class_names: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Overlay color-coded semantic masks on the original image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        label_map: (H, W) uint8 label map.
        alpha: Blending factor for overlay (0=image only, 1=mask only).
        class_names: Optional label-to-name mapping for legend.

    Returns:
        overlay: (H, W, 3) uint8 RGB image with colored mask overlay.
    """
    color_map = colorize_label_map(label_map, class_names)

    # Only overlay where labels > 0 (skip unlabeled background)
    mask_present = label_map > 0
    overlay = image.copy()
    overlay[mask_present] = (
        (1 - alpha) * image[mask_present].astype(np.float32)
        + alpha * color_map[mask_present].astype(np.float32)
    ).astype(np.uint8)

    return overlay


def visualize_frame(
    image_path: str,
    semantics_dir: str,
    frame_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional[np.ndarray]:
    """Create a comprehensive visualization for a single frame.

    Produces a 2×2 grid:
        [Original Image]  [Mask Overlay]
        [Label Map]        [Mask Count + Stats]

    Args:
        image_path: Path to original RGB image.
        semantics_dir: Path to the semantics/ output directory.
        frame_name: Frame identifier (e.g., "r_0").
        save_path: If provided, save the figure to this path.
        show: If True, display the figure (requires interactive backend).

    Returns:
        Composite image as numpy array if save_path is provided.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib is required for visualization. Install with: pip install matplotlib")
        return None

    # Load data
    image = np.array(Image.open(image_path).convert("RGB"))
    label_map = load_label_map(semantics_dir, frame_name)
    masks, mask_meta = load_masks(semantics_dir, frame_name)

    # Load metadata for class names
    metadata = load_metadata(semantics_dir)
    class_mapping = metadata.get("class_mapping", {})

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Panel 1: Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"Original: {frame_name}", fontsize=14)
    axes[0, 0].axis("off")

    # Panel 2: Mask overlay
    overlay = overlay_masks_on_image(image, label_map, alpha=0.5, class_names=class_mapping)
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"Semantic Overlay ({len(masks)} masks)", fontsize=14)
    axes[0, 1].axis("off")

    # Panel 3: Color-coded label map
    color_map = colorize_label_map(label_map, class_mapping)
    axes[1, 0].imshow(color_map)
    axes[1, 0].set_title("Semantic Label Map", fontsize=14)
    axes[1, 0].axis("off")

    # Add legend
    unique_labels = np.unique(label_map)
    legend_elements = []
    for label_id in unique_labels:
        color_idx = label_id % len(SEMANTIC_COLORS)
        color = SEMANTIC_COLORS[color_idx] / 255.0
        name = class_mapping.get(str(label_id), f"Class {label_id}")
        if label_id == 0:
            name = "unlabeled"
        legend_elements.append(
            patches.Patch(facecolor=color, edgecolor="black", label=name)
        )
    axes[1, 0].legend(
        handles=legend_elements, loc="lower right", fontsize=10,
        framealpha=0.8, facecolor="white",
    )

    # Panel 4: Statistics
    axes[1, 1].axis("off")
    stats_text = (
        f"Frame: {frame_name}\n"
        f"Image size: {image.shape[1]}×{image.shape[0]}\n"
        f"Masks generated: {len(masks)}\n"
        f"Unique labels: {len(unique_labels)}\n"
        f"Time: {mask_meta.get('frame_time', 'N/A')}\n\n"
        f"Per-class pixel coverage:\n"
    )
    total_pixels = label_map.shape[0] * label_map.shape[1]
    for label_id in unique_labels:
        count = np.sum(label_map == label_id)
        pct = 100.0 * count / total_pixels
        name = class_mapping.get(str(label_id), f"Class {label_id}")
        if label_id == 0:
            name = "unlabeled"
        stats_text += f"  {name}: {pct:.1f}%\n"

    axes[1, 1].text(
        0.1, 0.9, stats_text,
        transform=axes[1, 1].transAxes,
        fontsize=12, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    axes[1, 1].set_title("Extraction Statistics", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved visualization: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Convert figure to numpy array
    if save_path:
        return np.array(Image.open(save_path))
    return None


def visualize_clip_similarity(
    semantics_dir: str,
    frame_name: str,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Visualize per-pixel CLIP similarity to each text prompt as heatmaps.

    For each text prompt, shows a heatmap where brighter = higher cosine
    similarity between the pixel's CLIP feature and the text embedding.

    Args:
        semantics_dir: Path to semantics/ output directory.
        frame_name: Frame identifier.
        save_path: If provided, save the figure.
        show: If True, display the figure.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib required for visualization.")
        return

    # Load data
    feature_map = load_feature_map(semantics_dir, frame_name).astype(np.float32)
    text_features_path = Path(semantics_dir) / "clip_text_features.npy"
    if not text_features_path.exists():
        print("No text features found. Skipping similarity visualization.")
        return
    text_features = np.load(text_features_path)  # (K, D)

    metadata = load_metadata(semantics_dir)
    text_prompts = metadata.get("config", {}).get("text_prompts", [])

    H, W, D = feature_map.shape
    K = text_features.shape[0]

    # Compute per-pixel similarity to each text prompt
    # feature_map: (H, W, D), text_features: (K, D)
    # Reshape for batch matmul
    feat_flat = feature_map.reshape(-1, D)  # (H*W, D)
    # Normalize (features should already be normalized, but be safe)
    feat_norm = np.linalg.norm(feat_flat, axis=1, keepdims=True)
    feat_norm = np.maximum(feat_norm, 1e-8)
    feat_flat = feat_flat / feat_norm

    similarity = feat_flat @ text_features.T  # (H*W, K)
    similarity = similarity.reshape(H, W, K)

    # Create figure with one heatmap per prompt
    cols = min(4, K)
    rows = (K + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if K == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i in range(K):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        heatmap = similarity[:, :, i]

        im = ax.imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        prompt_name = text_prompts[i] if i < len(text_prompts) else f"Prompt {i}"
        ax.set_title(f'"{prompt_name}"\nmax={heatmap.max():.3f}', fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for i in range(K, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    plt.suptitle(f"CLIP Text Similarity Heatmaps — {frame_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved similarity heatmaps: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def create_semantic_video(
    dataset_path: str,
    semantics_dir: str,
    output_path: str,
    fps: int = 15,
    split: str = "train",
):
    """Compile semantic overlay frames into an MP4 video.

    Args:
        dataset_path: Root D-NeRF dataset directory.
        semantics_dir: Path to semantics/ output directory.
        output_path: Path to save the output MP4 video.
        fps: Frames per second for the output video.
        split: Which split to visualize ("train" or "test").
    """
    try:
        import imageio
    except ImportError:
        print("imageio required for video creation. Install: pip install imageio[ffmpeg]")
        return

    dataset_path = Path(dataset_path)
    semantics_dir_path = Path(semantics_dir)

    # Load transforms to get frame order
    transforms_file = dataset_path / f"transforms_{split}.json"
    if not transforms_file.exists():
        print(f"transforms_{split}.json not found in {dataset_path}")
        return

    with open(transforms_file, "r") as f:
        transforms = json.load(f)

    metadata = load_metadata(str(semantics_dir_path))
    class_mapping = metadata.get("class_mapping", {})

    frames_out = []
    for frame in transforms["frames"]:
        file_path = frame["file_path"]
        frame_name = Path(file_path).stem
        img_path = dataset_path / file_path
        if not img_path.suffix:
            img_path = img_path.with_suffix(".png")

        label_path = semantics_dir_path / "labels" / f"{frame_name}_labels.png"
        if not label_path.exists():
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        label_map = np.array(Image.open(label_path))
        overlay = overlay_masks_on_image(image, label_map, alpha=0.5, class_names=class_mapping)

        # Side-by-side: original | overlay
        combined = np.concatenate([image, overlay], axis=1)
        frames_out.append(combined)

    if frames_out:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(output_path), frames_out, fps=fps, quality=8)
        print(f"  ✓ Semantic video saved: {output_path} ({len(frames_out)} frames)")
    else:
        print("  ⚠ No frames found to compile into video.")


# ─────────────────────────────────────────────────────────────────────
# CLI Interface
# ─────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize semantic extraction results.",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to D-NeRF dataset root",
    )
    parser.add_argument(
        "--semantics_dir", type=str, default=None,
        help="Path to semantics/ dir (default: <dataset_path>/semantics/)",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Directory to save visualization outputs",
    )
    parser.add_argument(
        "--frame", type=str, default="r_0",
        help="Frame name to visualize (default: r_0)",
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Generate semantic overlay video",
    )
    parser.add_argument(
        "--similarity", action="store_true",
        help="Generate CLIP similarity heatmaps",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    semantics_dir = args.semantics_dir or str(dataset_path / "semantics")
    output_path = args.output_path or str(dataset_path / "semantics" / "visualizations")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Find the image path for the frame
    with open(dataset_path / "transforms_train.json", "r") as f:
        train_data = json.load(f)

    frame_path = None
    for frame in train_data["frames"]:
        if Path(frame["file_path"]).stem == args.frame:
            fp = dataset_path / frame["file_path"]
            if not fp.suffix:
                fp = fp.with_suffix(".png")
            frame_path = str(fp)
            break

    if frame_path is None:
        print(f"Frame '{args.frame}' not found in transforms_train.json")
        return

    # Single frame visualization
    print(f"Visualizing frame: {args.frame}")
    visualize_frame(
        image_path=frame_path,
        semantics_dir=semantics_dir,
        frame_name=args.frame,
        save_path=str(Path(output_path) / f"{args.frame}_visualization.png"),
    )

    # CLIP similarity heatmaps
    if args.similarity:
        print(f"Generating CLIP similarity heatmaps for: {args.frame}")
        visualize_clip_similarity(
            semantics_dir=semantics_dir,
            frame_name=args.frame,
            save_path=str(Path(output_path) / f"{args.frame}_clip_similarity.png"),
        )

    # Video
    if args.video:
        print("Generating semantic overlay video...")
        create_semantic_video(
            dataset_path=str(dataset_path),
            semantics_dir=semantics_dir,
            output_path=str(Path(output_path) / "semantic_overlay.mp4"),
        )


if __name__ == "__main__":
    main()
