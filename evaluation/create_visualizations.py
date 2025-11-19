"""
Visualization Tool for Method Comparison
=========================================
Creates visual comparisons of Custom Method vs DiffEdit results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path
import json
import numpy as np
from typing import List, Optional
import argparse


def create_comparison_grid(
    original_path: Path,
    custom_path: Path,
    diffedit_path: Path,
    custom_mask_path: Optional[Path] = None,
    diffedit_mask_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None
):
    """
    Create a comparison grid showing original and both methods
    
    Args:
        original_path: Path to original image
        custom_path: Path to custom method output
        diffedit_path: Path to DiffEdit output
        custom_mask_path: Path to custom method mask (optional)
        diffedit_mask_path: Path to DiffEdit mask (optional)
        metrics: Dictionary with per-image metrics (optional)
        output_path: Where to save the comparison (optional)
        title: Title for the comparison (optional)
    """
    
    # Load images
    original = Image.open(original_path).convert("RGB")
    custom = Image.open(custom_path).convert("RGB")
    diffedit = Image.open(diffedit_path).convert("RGB")
    
    # Determine grid size
    has_masks = custom_mask_path and diffedit_mask_path
    n_cols = 5 if has_masks else 3
    
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Custom method
    axes[1].imshow(custom)
    title_custom = "Custom Method"
    if metrics and 'custom' in metrics:
        custom_metrics = metrics['custom']
        title_custom += f"\nFID: {custom_metrics.get('fid', 'N/A'):.2f}"
    axes[1].set_title(title_custom, fontsize=14, fontweight='bold', color='green')
    axes[1].axis('off')
    
    # DiffEdit
    axes[2].imshow(diffedit)
    title_diffedit = "DiffEdit"
    if metrics and 'diffedit' in metrics:
        diffedit_metrics = metrics['diffedit']
        title_diffedit += f"\nFID: {diffedit_metrics.get('fid', 'N/A'):.2f}"
    axes[2].set_title(title_diffedit, fontsize=14, fontweight='bold', color='blue')
    axes[2].axis('off')
    
    # Masks if available
    if has_masks and custom_mask_path and diffedit_mask_path:
        custom_mask = Image.open(custom_mask_path).convert("L")
        diffedit_mask = Image.open(diffedit_mask_path).convert("L")
        
        axes[3].imshow(custom_mask, cmap='gray')
        axes[3].set_title("Custom Mask", fontsize=12)
        axes[3].axis('off')
        
        axes[4].imshow(diffedit_mask, cmap='gray')
        axes[4].set_title("DiffEdit Mask", fontsize=12)
        axes[4].axis('off')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_metrics_comparison_chart(
    custom_metrics: dict,
    diffedit_metrics: dict,
    output_path: Optional[Path] = None
):
    """
    Create a bar chart comparing metrics side-by-side
    """
    
    metrics_to_plot = {
        'FID': ('fid', False),  # False = lower is better
        'IS': ('inception_score_mean', True),  # True = higher is better
        'LPIPS': ('lpips', False),
        'CLIP': ('clip', True),
        'IoU': ('iou', True)
    }
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))
    
    for idx, (metric_name, (metric_key, higher_better)) in enumerate(metrics_to_plot.items()):
        custom_val = custom_metrics.get(metric_key)
        diffedit_val = diffedit_metrics.get(metric_key)
        
        if custom_val is None or diffedit_val is None:
            axes[idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[idx].set_title(metric_name)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            continue
        
        # Determine winner
        if higher_better:
            winner = 'custom' if custom_val > diffedit_val else 'diffedit'
        else:
            winner = 'custom' if custom_val < diffedit_val else 'diffedit'
        
        # Plot bars
        colors = ['green' if winner == 'custom' else 'lightgreen',
                 'blue' if winner == 'diffedit' else 'lightblue']
        
        bars = axes[idx].bar(['Custom', 'DiffEdit'], 
                            [custom_val, diffedit_val],
                            color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=10)
        
        # Styling
        direction = '↑' if higher_better else '↓'
        axes[idx].set_title(f'{metric_name} {direction}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Metrics Comparison: Custom Method vs DiffEdit', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved metrics chart to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_all_comparisons(
    results_json: Path,
    original_dir: Path,
    custom_dir: Path,
    diffedit_dir: Path,
    custom_masks_dir: Optional[Path] = None,
    diffedit_masks_dir: Optional[Path] = None,
    output_dir: Path = Path("visualizations"),
    max_images: int = 10
):
    """
    Create comparison visualizations for all images
    """
    
    # Load results
    with open(results_json, 'r') as f:
        results = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create metrics comparison chart
    create_metrics_comparison_chart(
        custom_metrics=results['custom_method'],
        diffedit_metrics=results['diffedit_method'],
        output_path=output_dir / "metrics_comparison.png"
    )
    
    # Create per-image comparisons
    per_image = results.get('per_image_comparison', [])[:max_images]
    
    print(f"\nCreating {len(per_image)} image comparisons...")
    
    for i, img_data in enumerate(per_image, 1):
        img_name = img_data['image']
        
        # Find image paths
        original_path = original_dir / img_name
        custom_path = custom_dir / img_name
        diffedit_path = diffedit_dir / f"edited_{img_name}"
        
        # Alternative paths
        if not custom_path.exists():
            custom_path = custom_dir / f"final_{img_name}"
        if not diffedit_path.exists():
            diffedit_path = diffedit_dir / img_name
        
        # Check if all paths exist
        if not all([p.exists() for p in [original_path, custom_path, diffedit_path]]):
            print(f"⚠ Skipping {img_name} - missing files")
            continue
        
        # Find mask paths
        custom_mask = None
        diffedit_mask = None
        
        if custom_masks_dir:
            custom_mask = custom_masks_dir / img_name
            if not custom_mask.exists():
                custom_mask = custom_masks_dir / f"stage3_mask_{img_name}"
        
        if diffedit_masks_dir:
            diffedit_mask = diffedit_masks_dir / f"mask_{img_name}"
            if not diffedit_mask.exists():
                diffedit_mask = diffedit_masks_dir / img_name
        
        # Create comparison
        output_path = output_dir / f"comparison_{i:03d}_{img_name}"
        
        metrics_dict = {
            'custom': img_data['custom'],
            'diffedit': img_data['diffedit']
        }
        
        create_comparison_grid(
            original_path=original_path,
            custom_path=custom_path,
            diffedit_path=diffedit_path,
            custom_mask_path=custom_mask if custom_mask and custom_mask.exists() else None,
            diffedit_mask_path=diffedit_mask if diffedit_mask and diffedit_mask.exists() else None,
            metrics=metrics_dict,
            output_path=output_path,
            title=f"Comparison {i}: {img_name}"
        )
    
    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"✓ Created metrics chart: {output_dir / 'metrics_comparison.png'}")
    print(f"✓ Created {len(per_image)} image comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visual comparisons")
    parser.add_argument("--results_json", type=str, required=True,
                       help="Path to comparison_results.json")
    parser.add_argument("--original_dir", type=str, required=True,
                       help="Directory with original images")
    parser.add_argument("--custom_dir", type=str, required=True,
                       help="Directory with custom method outputs")
    parser.add_argument("--diffedit_dir", type=str, required=True,
                       help="Directory with DiffEdit outputs")
    parser.add_argument("--custom_masks", type=str, default=None,
                       help="Directory with custom method masks")
    parser.add_argument("--diffedit_masks", type=str, default=None,
                       help="Directory with DiffEdit masks")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--max_images", type=int, default=10,
                       help="Maximum number of image comparisons to create")
    
    args = parser.parse_args()
    
    create_all_comparisons(
        results_json=Path(args.results_json),
        original_dir=Path(args.original_dir),
        custom_dir=Path(args.custom_dir),
        diffedit_dir=Path(args.diffedit_dir),
        custom_masks_dir=Path(args.custom_masks) if args.custom_masks else None,
        diffedit_masks_dir=Path(args.diffedit_masks) if args.diffedit_masks else None,
        output_dir=Path(args.output_dir),
        max_images=args.max_images
    )
    
    print("\n✅ Visualization complete!")
