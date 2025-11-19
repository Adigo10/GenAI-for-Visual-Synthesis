"""
Comprehensive Evaluation Script: Custom Method vs DiffEdit
============================================================
This script evaluates and compares:
1. Custom Method (UNet Segmentation + Stable Diffusion Inpainting)
2. DiffEdit Method (from HuggingFace implementation)

Metrics computed:
- FID (Fréchet Inception Distance): Image quality and diversity
- IS (Inception Score): Image quality and diversity
- IoU (Intersection over Union): Mask/segmentation accuracy
- LPIPS (Learned Perceptual Image Patch Similarity): Perceptual similarity
- CLIP Score: Text-image alignment
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Import evaluation utilities
from evaluate import MetricsEvaluator, print_results

# For Inception Score calculation
from torchvision.models import inception_v3
import torch.nn.functional as F
from scipy.stats import entropy


class InceptionScore:
    """Calculate Inception Score for generated images"""
    
    def __init__(self, device="cuda", batch_size=32, splits=10):
        self.device = device
        self.batch_size = batch_size
        self.splits = splits
        
        # Load Inception v3 model
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        
        print("✓ Inception Score model initialized!")
    
    def _get_predictions(self, images: List[Image.Image]) -> np.ndarray:
        """Get predictions from Inception v3 for a list of images"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        preds = []
        for i in tqdm(range(0, len(images), self.batch_size), desc="Computing predictions"):
            batch = images[i:i + self.batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch]).to(self.device)
            
            with torch.no_grad():
                pred = self.model(batch_tensors)
                pred = F.softmax(pred, dim=1).cpu().numpy()
            
            preds.append(pred)
        
        return np.concatenate(preds, axis=0)
    
    def calculate(self, images: List[Image.Image]) -> Tuple[float, float]:
        """
        Calculate Inception Score
        
        Returns:
            mean_score: Mean IS across splits
            std_score: Standard deviation of IS across splits
        """
        preds = self._get_predictions(images)
        
        # Split predictions into groups
        split_scores = []
        n = len(preds)
        split_size = n // self.splits
        
        for k in range(self.splits):
            start = k * split_size
            end = (k + 1) * split_size if k < self.splits - 1 else n
            part = preds[start:end]
            
            # Calculate p(y)
            py = np.mean(part, axis=0)
            
            # Calculate KL divergence for each image
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            
            split_scores.append(np.exp(np.mean(scores)))
        
        return np.mean(split_scores), np.std(split_scores)


class MethodComparator:
    """Compare Custom Method vs DiffEdit across multiple metrics"""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.evaluator = MetricsEvaluator(device=device)
        self.inception_scorer = InceptionScore(device=device)
        
    def evaluate_method(
        self,
        method_name: str,
        original_images_dir: Path,
        generated_images_dir: Path,
        generated_masks_dir: Optional[Path] = None,
        ground_truth_masks_dir: Optional[Path] = None,
        prompts_dict: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """
        Evaluate a single method (Custom or DiffEdit)
        
        Args:
            method_name: Name of the method being evaluated
            original_images_dir: Directory with original test images
            generated_images_dir: Directory with generated results
            generated_masks_dir: Directory with generated masks (optional)
            ground_truth_masks_dir: Directory with ground truth masks (optional)
            prompts_dict: Dictionary mapping image names to prompts
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {method_name}")
        print(f"{'='*70}\n")
        
        results = {
            'method': method_name,
            'fid': None,
            'inception_score_mean': None,
            'inception_score_std': None,
            'lpips_scores': [],
            'clip_scores': [],
            'iou_scores': [],
            'per_image_results': []
        }
        
        # Get list of images
        original_images_dir = Path(original_images_dir)
        generated_images_dir = Path(generated_images_dir)
        
        image_files = sorted([
            f for f in original_images_dir.iterdir() 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        if not image_files:
            print(f"❌ No images found in {original_images_dir}")
            return results
        
        print(f"Found {len(image_files)} images to evaluate")
        
        # Load all images for Inception Score
        generated_images = []
        
        for img_path in tqdm(image_files, desc=f"Processing {method_name}"):
            img_name = img_path.name
            
            # Construct generated image path
            gen_img_path = generated_images_dir / img_name
            if not gen_img_path.exists():
                # Try alternative naming conventions
                alt_names = [
                    f"edited_{img_name}",
                    f"final_{img_name}",
                    f"stage4_{img_name}",
                    f"output_{img_name}"
                ]
                for alt_name in alt_names:
                    alt_path = generated_images_dir / alt_name
                    if alt_path.exists():
                        gen_img_path = alt_path
                        break
                else:
                    print(f"⚠ Generated image not found for: {img_name}")
                    continue
            
            # Load images
            orig_img = Image.open(img_path).convert("RGB")
            gen_img = Image.open(gen_img_path).convert("RGB")
            
            generated_images.append(gen_img)
            
            # 1. Update FID (accumulate features)
            orig_tensor = self.evaluator.preprocess_for_fid(orig_img)
            gen_tensor = self.evaluator.preprocess_for_fid(gen_img)
            self.evaluator.fid.update(orig_tensor, real=True)
            self.evaluator.fid.update(gen_tensor, real=False)
            
            # 2. Calculate LPIPS
            orig_lpips = self.evaluator.preprocess_for_lpips(orig_img)
            gen_lpips = self.evaluator.preprocess_for_lpips(gen_img)
            lpips_score = self.evaluator.lpips(gen_lpips, orig_lpips).item()
            results['lpips_scores'].append(lpips_score)
            
            # 3. Calculate CLIP similarity (if prompt available)
            clip_score = None
            if prompts_dict and img_name in prompts_dict:
                prompt = prompts_dict[img_name]
                clip_score = self.evaluator.calculate_clip_similarity(gen_img, prompt)
                results['clip_scores'].append(clip_score)
            
            # 4. Calculate IoU (if masks available)
            iou_score = None
            if generated_masks_dir and ground_truth_masks_dir:
                gen_mask_path = generated_masks_dir / img_name
                gt_mask_path = ground_truth_masks_dir / img_name
                
                # Try alternative mask naming
                if not gen_mask_path.exists():
                    alt_mask_names = [
                        f"mask_{img_name}",
                        f"stage1_{img_name}",
                        f"stage3_{img_name}"
                    ]
                    for alt_name in alt_mask_names:
                        alt_path = generated_masks_dir / alt_name
                        if alt_path.exists():
                            gen_mask_path = alt_path
                            break
                
                if gen_mask_path.exists() and gt_mask_path.exists():
                    gt_mask = Image.open(gt_mask_path).convert("L")
                    gen_mask = Image.open(gen_mask_path).convert("L")
                    iou_score = self.evaluator.calculate_segmentation_iou(gen_mask, gt_mask)
                    results['iou_scores'].append(iou_score)
            
            # Store per-image results
            results['per_image_results'].append({
                'image': img_name,
                'lpips': lpips_score,
                'clip': clip_score,
                'iou': iou_score
            })
        
        # Compute final FID
        print("\nComputing FID score...")
        results['fid'] = self.evaluator.fid.compute().item()
        
        # Compute Inception Score
        if generated_images:
            print("Computing Inception Score...")
            is_mean, is_std = self.inception_scorer.calculate(generated_images)
            results['inception_score_mean'] = is_mean
            results['inception_score_std'] = is_std
        
        # Calculate averages
        results['avg_lpips'] = np.mean(results['lpips_scores']) if results['lpips_scores'] else None
        results['avg_clip'] = np.mean(results['clip_scores']) if results['clip_scores'] else None
        results['avg_iou'] = np.mean(results['iou_scores']) if results['iou_scores'] else None
        
        return results
    
    def print_comparison(self, custom_results: Dict, diffedit_results: Dict):
        """Print side-by-side comparison of both methods"""
        
        print(f"\n{'='*70}")
        print("COMPARATIVE RESULTS: Custom Method vs DiffEdit")
        print(f"{'='*70}\n")
        
        # Create comparison table
        metrics = [
            ("FID Score", "fid", "↓", "lower"),
            ("Inception Score", "inception_score_mean", "↑", "higher"),
            ("LPIPS", "avg_lpips", "↓", "lower"),
            ("CLIP Similarity", "avg_clip", "↑", "higher"),
            ("IoU (Segmentation)", "avg_iou", "↑", "higher")
        ]
        
        print(f"{'Metric':<25} {'Custom Method':<18} {'DiffEdit':<18} {'Winner':<15}")
        print("-" * 70)
        
        winners = {'custom': 0, 'diffedit': 0, 'tie': 0}
        
        for metric_name, metric_key, direction, better in metrics:
            custom_val = custom_results.get(metric_key)
            diffedit_val = diffedit_results.get(metric_key)
            
            if custom_val is None or diffedit_val is None:
                custom_str = "N/A" if custom_val is None else f"{custom_val:.4f}"
                diffedit_str = "N/A" if diffedit_val is None else f"{diffedit_val:.4f}"
                winner = "N/A"
            else:
                custom_str = f"{custom_val:.4f}"
                diffedit_str = f"{diffedit_val:.4f}"
                
                if better == "lower":
                    if custom_val < diffedit_val:
                        winner = "✓ Custom"
                        winners['custom'] += 1
                    elif custom_val > diffedit_val:
                        winner = "✓ DiffEdit"
                        winners['diffedit'] += 1
                    else:
                        winner = "Tie"
                        winners['tie'] += 1
                else:  # higher is better
                    if custom_val > diffedit_val:
                        winner = "✓ Custom"
                        winners['custom'] += 1
                    elif custom_val < diffedit_val:
                        winner = "✓ DiffEdit"
                        winners['diffedit'] += 1
                    else:
                        winner = "Tie"
                        winners['tie'] += 1
            
            print(f"{metric_name:<25} {custom_str:<18} {diffedit_str:<18} {winner:<15}")
        
        print("-" * 70)
        print(f"\nOVERALL WINNER: ", end="")
        if winners['custom'] > winners['diffedit']:
            print("✓ Custom Method")
        elif winners['diffedit'] > winners['custom']:
            print("✓ DiffEdit")
        else:
            print("Tie")
        
        print(f"(Custom wins: {winners['custom']}, DiffEdit wins: {winners['diffedit']}, Ties: {winners['tie']})")
        print(f"\n{'='*70}\n")


def load_prompts_from_file(prompts_file: Path) -> Dict[str, str]:
    """
    Load prompts from a JSON or text file
    
    Expected JSON format:
    {
        "image1.jpg": "prompt for image 1",
        "image2.jpg": "prompt for image 2"
    }
    """
    prompts_file = Path(prompts_file)
    
    if prompts_file.suffix == '.json':
        with open(prompts_file, 'r') as f:
            return json.load(f)
    else:
        # Parse text file format: "filename.jpg: prompt text"
        prompts = {}
        with open(prompts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    filename, prompt = line.split(':', 1)
                    prompts[filename.strip()] = prompt.strip()
        return prompts


def save_results_to_file(custom_results: Dict, diffedit_results: Dict, output_path: Path):
    """Save comparison results to JSON file"""
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'custom_method': {
            'fid': custom_results.get('fid'),
            'inception_score_mean': custom_results.get('inception_score_mean'),
            'inception_score_std': custom_results.get('inception_score_std'),
            'lpips': custom_results.get('avg_lpips'),
            'clip': custom_results.get('avg_clip'),
            'iou': custom_results.get('avg_iou'),
            'num_images': len(custom_results.get('per_image_results', []))
        },
        'diffedit_method': {
            'fid': diffedit_results.get('fid'),
            'inception_score_mean': diffedit_results.get('inception_score_mean'),
            'inception_score_std': diffedit_results.get('inception_score_std'),
            'lpips': diffedit_results.get('avg_lpips'),
            'clip': diffedit_results.get('avg_clip'),
            'iou': diffedit_results.get('avg_iou'),
            'num_images': len(diffedit_results.get('per_image_results', []))
        },
        'per_image_comparison': []
    }
    
    # Add per-image comparison
    for custom_img, diffedit_img in zip(
        custom_results.get('per_image_results', []),
        diffedit_results.get('per_image_results', [])
    ):
        comparison['per_image_comparison'].append({
            'image': custom_img['image'],
            'custom': {
                'lpips': custom_img['lpips'],
                'clip': custom_img['clip'],
                'iou': custom_img['iou']
            },
            'diffedit': {
                'lpips': diffedit_img['lpips'],
                'clip': diffedit_img['clip'],
                'iou': diffedit_img['iou']
            }
        })
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"✓ Results saved to: {output_path}")


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Custom Method vs DiffEdit")
    parser.add_argument("--original_images", type=str, required=True,
                       help="Directory with original test images")
    parser.add_argument("--custom_outputs", type=str, required=True,
                       help="Directory with custom method outputs")
    parser.add_argument("--diffedit_outputs", type=str, required=True,
                       help="Directory with DiffEdit outputs")
    parser.add_argument("--custom_masks", type=str, default=None,
                       help="Directory with custom method masks (optional)")
    parser.add_argument("--diffedit_masks", type=str, default=None,
                       help="Directory with DiffEdit masks (optional)")
    parser.add_argument("--ground_truth_masks", type=str, default=None,
                       help="Directory with ground truth masks (optional)")
    parser.add_argument("--prompts", type=str, default=None,
                       help="JSON/text file with prompts for each image")
    parser.add_argument("--output_json", type=str, default="comparison_results.json",
                       help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Load prompts if provided
    prompts_dict = None
    if args.prompts:
        prompts_dict = load_prompts_from_file(Path(args.prompts))
        print(f"✓ Loaded {len(prompts_dict)} prompts from {args.prompts}")
    
    # Initialize comparator
    comparator = MethodComparator(device=args.device)
    
    # Evaluate Custom Method
    custom_results = comparator.evaluate_method(
        method_name="Custom Method (UNet + SD Inpainting)",
        original_images_dir=Path(args.original_images),
        generated_images_dir=Path(args.custom_outputs),
        generated_masks_dir=Path(args.custom_masks) if args.custom_masks else None,
        ground_truth_masks_dir=Path(args.ground_truth_masks) if args.ground_truth_masks else None,
        prompts_dict=prompts_dict
    )
    
    # Reset FID for next evaluation
    comparator.evaluator.fid.reset()
    
    # Evaluate DiffEdit
    diffedit_results = comparator.evaluate_method(
        method_name="DiffEdit (HuggingFace)",
        original_images_dir=Path(args.original_images),
        generated_images_dir=Path(args.diffedit_outputs),
        generated_masks_dir=Path(args.diffedit_masks) if args.diffedit_masks else None,
        ground_truth_masks_dir=Path(args.ground_truth_masks) if args.ground_truth_masks else None,
        prompts_dict=prompts_dict
    )
    
    # Print comparison
    comparator.print_comparison(custom_results, diffedit_results)
    
    # Save results
    save_results_to_file(custom_results, diffedit_results, Path(args.output_json))
    
    print("\n✅ Evaluation complete!")
