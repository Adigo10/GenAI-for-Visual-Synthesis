import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import clip
from tqdm import tqdm
import os

# --------------------------- SETUP --------------------------- #
class MetricsEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        
        # FID - uses Inception v3
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        
        # LPIPS - perceptual similarity
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        
        # CLIP - semantic similarity
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        print("✓ All metrics initialized!")
    
    def preprocess_for_fid(self, image):
        """Convert PIL image to tensor for FID (expects [0,255] uint8)"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # FID expects [B, 3, H, W] in range [0, 255]
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def preprocess_for_lpips(self, image):
        """Convert PIL image to tensor for LPIPS (expects [-1,1])"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)
    
    def calculate_clip_similarity(self, image, text_prompt):
        """Calculate CLIP image-text similarity"""
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([text_prompt]).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(text_input)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate cosine similarity
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    def calculate_segmentation_iou(self, pred_mask, gt_mask):
        """Calculate IoU between predicted and ground truth masks"""
        # Convert to binary
        pred_binary = (np.array(pred_mask) > 127).astype(np.float32)
        gt_binary = (np.array(gt_mask) > 127).astype(np.float32)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_and(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 0.0
        return intersection / union

# --------------------------- BATCH EVALUATION --------------------------- #
def evaluate_model(
    evaluator,
    test_images_dir,
    generated_images_dir,
    prompts_dict=None,  # {image_name: prompt}
    gt_masks_dir=None,  # Optional: if you have ground truth masks
    generated_masks_dir=None  # Your generated masks
):
    """
    Evaluate a set of generated images against originals
    
    Args:
        test_images_dir: Directory with original test images
        generated_images_dir: Directory with your generated results
        prompts_dict: Dictionary mapping image names to prompts used
        gt_masks_dir: Optional ground truth masks for IoU calculation
        generated_masks_dir: Your generated masks
    """
    
    results = {
        'fid': None,
        'lpips_scores': [],
        'clip_scores': [],
        'iou_scores': [],
        'per_image_results': []
    }
    
    # Get list of test images
    test_images = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"\n{'='*60}")
    print(f"Evaluating {len(test_images)} images...")
    print(f"{'='*60}\n")
    
    for img_name in tqdm(test_images, desc="Processing images"):
        # Load images
        orig_path = os.path.join(test_images_dir, img_name)
        gen_path = os.path.join(generated_images_dir, f"edited_{img_name}")
        
        if not os.path.exists(gen_path):
            print(f"⚠ Generated image not found: {gen_path}")
            continue
        
        orig_img = Image.open(orig_path).convert("RGB")
        gen_img = Image.open(gen_path).convert("RGB")
        
        # 1. Update FID (accumulate features)
        orig_tensor = evaluator.preprocess_for_fid(orig_img)
        gen_tensor = evaluator.preprocess_for_fid(gen_img)
        evaluator.fid.update(orig_tensor, real=True)
        evaluator.fid.update(gen_tensor, real=False)
        
        # 2. Calculate LPIPS
        orig_lpips = evaluator.preprocess_for_lpips(orig_img)
        gen_lpips = evaluator.preprocess_for_lpips(gen_img)
        lpips_score = evaluator.lpips(gen_lpips, orig_lpips).item()
        results['lpips_scores'].append(lpips_score)
        
        # 3. Calculate CLIP similarity (if prompt available)
        clip_score = None
        if prompts_dict and img_name in prompts_dict:
            prompt = prompts_dict[img_name]
            clip_score = evaluator.calculate_clip_similarity(gen_img, prompt)
            results['clip_scores'].append(clip_score)
        
        # 4. Calculate IoU (if masks available)
        iou_score = None
        if gt_masks_dir and generated_masks_dir:
            gt_mask_path = os.path.join(gt_masks_dir, img_name)
            gen_mask_path = os.path.join(generated_masks_dir, f"mask_{img_name}")
            
            if os.path.exists(gt_mask_path) and os.path.exists(gen_mask_path):
                gt_mask = Image.open(gt_mask_path).convert("L")
                gen_mask = Image.open(gen_mask_path).convert("L")
                iou_score = evaluator.calculate_segmentation_iou(gen_mask, gt_mask)
                results['iou_scores'].append(iou_score)
        
        # Store per-image results
        results['per_image_results'].append({
            'image': img_name,
            'lpips': lpips_score,
            'clip': clip_score,
            'iou': iou_score
        })
    
    # Compute final FID
    results['fid'] = evaluator.fid.compute().item()
    
    # Calculate averages
    results['avg_lpips'] = np.mean(results['lpips_scores']) if results['lpips_scores'] else None
    results['avg_clip'] = np.mean(results['clip_scores']) if results['clip_scores'] else None
    results['avg_iou'] = np.mean(results['iou_scores']) if results['iou_scores'] else None
    
    return results

# --------------------------- PRINT RESULTS --------------------------- #
def print_results(results, method_name="Model"):
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {method_name}")
    print(f"{'='*60}")
    print(f"FID Score:        {results['fid']:.3f} ↓ (lower is better)")
    print(f"LPIPS Score:      {results['avg_lpips']:.4f} ↓ (lower is better)")
    if results['avg_clip']:
        print(f"CLIP Similarity:  {results['avg_clip']:.4f} ↑ (higher is better)")
    if results['avg_iou']:
        print(f"Segmentation IoU: {results['avg_iou']:.4f} ↑ (higher is better)")
    print(f"{'='*60}\n")

# --------------------------- USAGE EXAMPLE --------------------------- #
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = MetricsEvaluator(device="cuda")
    
    # Example: Evaluate COCO model
    prompts = {
        "test_001.jpg": "a red sports car on a road",
        "test_002.jpg": "a blue sedan in a parking lot",
        # Add all your test prompts here
    }
    
    results_coco = evaluate_model(
        evaluator=evaluator,
        test_images_dir="data/test_images",
        generated_images_dir="outputs/coco_model",
        prompts_dict=prompts,
        gt_masks_dir="data/test_masks",  # Optional
        generated_masks_dir="outputs/coco_model"
    )
    
    print_results(results_coco, "COCO Model")
    
    # Reset FID for next evaluation
    evaluator.fid.reset()
    
    # Evaluate Carvana model
    results_carvana = evaluate_model(
        evaluator=evaluator,
        test_images_dir="data/test_images",
        generated_images_dir="outputs/carvana_model",
        prompts_dict=prompts,
        gt_masks_dir="data/test_masks",
        generated_masks_dir="outputs/carvana_model"
    )
    
    print_results(results_carvana, "Carvana Model")
    
    # Save results to JSON
    import json
    
    comparison = {
        "coco_model": {
            "fid": results_coco['fid'],
            "lpips": results_coco['avg_lpips'],
            "clip": results_coco['avg_clip'],
            "iou": results_coco['avg_iou']
        },
        "carvana_model": {
            "fid": results_carvana['fid'],
            "lpips": results_carvana['avg_lpips'],
            "clip": results_carvana['avg_clip'],
            "iou": results_carvana['avg_iou']
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("✓ Results saved to evaluation_results.json")