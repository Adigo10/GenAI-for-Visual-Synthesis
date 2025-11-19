"""
Batch Processing Script for Custom Method (Two-Stage Pipeline)
================================================================
Runs the custom UNet + Stable Diffusion pipeline (Stage 1 + 2) on multiple test images.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm import tqdm
import shutil
import torch
from diffusers import StableDiffusionInpaintPipeline

# Import custom pipeline functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import (
    segment_image,
    regenerate_vehicle,
    set_global_pipeline,
    BASE_DIR
)


def process_image_with_custom_method(
    img_path: Path,
    unet_path: Path,
    source_subject: str,
    target_subject: str,
    output_dir: Path,
    device: str = "cuda",
    vehicle_negative_prompt: str = "blurry, low quality, distorted, ugly car, deformed vehicle"
):
    """
    Process a single image through the custom 2-stage pipeline
    
    Args:
        img_path: Path to input image
        unet_path: Path to UNet segmentation model
        source_subject: Description of source vehicle
        target_subject: Description of target vehicle
        output_dir: Directory to save all outputs
        device: Device to use ('cuda', 'mps', 'cpu')
        vehicle_negative_prompt: Negative prompt for vehicle generation
    
    Returns:
        Dictionary with paths to all generated outputs
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name = img_path.name
    results = {
        'input': str(img_path),
        'stage1_mask': None,
        'stage2_vehicle': None
    }
    
    try:
        # Stage 1: Vehicle Segmentation
        stage1_mask_name = f"stage1_mask_{img_name}"
        stage1_mask_path = segment_image(
            img_path=str(img_path),
            model_path=str(unet_path),
            output_dir=output_dir,
            output_name=stage1_mask_name,
            device=device
        )
        results['stage1_mask'] = stage1_mask_path
        
        # Stage 2: Vehicle Inpainting (uses global pipeline)
        stage2_vehicle_name = f"stage2_vehicle_{img_name}"
        _, stage2_vehicle_path = regenerate_vehicle(
            img_path=str(img_path),
            mask_path=stage1_mask_path,
            model_dir=None,  # Uses global pipeline
            target_prompt=target_subject,
            source_prompt=source_subject,
            negative_prompt=vehicle_negative_prompt,
            output_dir=output_dir,
            output_name=stage2_vehicle_name,
            device=device
        )
        results['stage2_vehicle'] = stage2_vehicle_path
        results['status'] = 'success'
        
    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results


def batch_process_custom_method(
    input_dir: Path,
    output_dir: Path,
    prompts_dict: Dict,
    unet_path: Optional[Path] = None
):
    """
    Process multiple images with custom two-stage method
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save all outputs
        prompts_dict: Dictionary with unified prompts per image
            Format: {
                "image1.jpg": {
                    "source_subject": "a car",
                    "target_subject": "a sleek modern sports car",
                    "original_caption": "a car on the road"
                },
                ...
            }
        unet_path: Path to UNet model (defaults to model/unet_model_carvana_new.pth)
    """
    
    # Set default model paths if not provided
    if unet_path is None:
        unet_path = BASE_DIR / "model" / "unet_model_carvana_new.pth"
    
    # Validate model paths
    if not unet_path.exists():
        raise FileNotFoundError(f"UNet model not found: {unet_path}")
    
    # Initialize device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print("🚀 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        dtype = torch.float32
        print("⚠️  Using CPU (this will be slow!)")
    
    # Load global pipeline once for all images
    print("\n🚀 Loading Stable Diffusion pipeline...")
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)
    set_global_pipeline(pipeline, device)
    print("✓ Pipeline loaded successfully!")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    print(f"\nProcessing {len(image_files)} images with Custom Two-Stage Method...")
    print(f"{'='*60}\n")
    print(f"UNet model: {unet_path}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}\n")
    
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images (2 Stages)"):
        img_name = img_path.name
        
        # Get unified prompts for this image
        if img_name not in prompts_dict:
            print(f"⚠ No prompts found for {img_name}, skipping...")
            continue
        
        prompts = prompts_dict[img_name]
        source_subject = prompts.get('source_subject', 'a car')
        target_subject = prompts.get('target_subject', 'a sleek modern sports car')
        original_caption = prompts.get('original_caption', '')
        
        if not source_subject or not target_subject:
            print(f"⚠ Missing source or target subject for {img_name}, skipping...")
            continue
        
        # Process image (2 stages only)
        results = process_image_with_custom_method(
            img_path=img_path,
            unet_path=unet_path,
            source_subject=source_subject,
            target_subject=target_subject,
            output_dir=output_dir,
            device=device
        )
        
        results['source_subject'] = source_subject
        results['target_subject'] = target_subject
        results['original_caption'] = original_caption
        all_results.append(results)
    
    # Save processing log
    log_path = output_dir / "processing_log.json"
    with open(log_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create organized output folders
    organize_outputs(output_dir, all_results)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Processed {len([r for r in all_results if r.get('status') == 'success'])} images successfully")
    print(f"✓ Log saved to: {log_path}")
    
    return all_results


def organize_outputs(output_dir: Path, results: list):
    """
    Organize outputs into separate folders for easier evaluation
    Creates:
    - final_images/ - Stage 2 vehicle outputs (final result)
    - masks/ - Stage 1 masks
    """
    
    final_dir = output_dir / "final_images"
    masks_dir = output_dir / "masks"
    
    final_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    
    for result in results:
        if result.get('status') != 'success':
            continue
        
        # Copy Stage 2 vehicle image as final output
        if result['stage2_vehicle']:
            src = Path(result['stage2_vehicle'])
            dst = final_dir / src.name.replace('stage2_vehicle_', '')
            if src.exists():
                shutil.copy2(src, dst)
        
        # Copy Stage 1 mask
        if result['stage1_mask']:
            src = Path(result['stage1_mask'])
            dst = masks_dir / src.name.replace('stage1_mask_', '')
            if src.exists():
                shutil.copy2(src, dst)
    
    print(f"\n✓ Outputs organized:")
    print(f"  - Final images: {final_dir}")
    print(f"  - Masks: {masks_dir}")


def load_prompts_from_file(prompts_file: Path) -> dict:
    """
    Load unified prompts from JSON file
    
    Expected format:
    {
        "image1.jpg": {
            "source_subject": "a red car",
            "target_subject": "a sleek red sports car",
            "original_caption": "a red car on the road"
        },
        "image2.jpg": {
            "source_subject": "a blue sedan",
            "target_subject": "a modern blue luxury sedan",
            "original_caption": "a blue sedan parked"
        }
    }
    """
    with open(prompts_file, 'r') as f:
        return json.load(f)


def create_sample_prompts_file(output_path: Path):
    """Create a sample unified prompts file for reference"""
    sample_prompts = {
        "example1.jpg": {
            "source_subject": "a red car",
            "target_subject": "a sleek red sports car with glossy finish",
            "original_caption": "a red car on the road"
        },
        "example2.jpg": {
            "source_subject": "a blue sedan",
            "target_subject": "a modern blue luxury sedan",
            "original_caption": "a blue sedan parked on street"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_prompts, f, indent=2)
    
    print(f"✓ Sample unified prompts file created: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Custom Two-Stage Method on batch of images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save custom method outputs")
    parser.add_argument("--prompts", type=str, required=True,
                       help="JSON file with unified prompts (source_subject/target_subject) per image")
    parser.add_argument("--unet_path", type=str, default=None,
                       help="Path to UNet model (optional)")
    parser.add_argument("--create_sample_prompts", action="store_true",
                       help="Create a sample unified prompts file and exit")
    
    args = parser.parse_args()
    
    # Create sample prompts file if requested
    if args.create_sample_prompts:
        create_sample_prompts_file(Path("sample_unified_prompts.json"))
        exit(0)
    
    # Load unified prompts
    prompts_dict = load_prompts_from_file(Path(args.prompts))
    print(f"✓ Loaded unified prompts for {len(prompts_dict)} images")
    
    # Parse optional model path
    unet_path = Path(args.unet_path) if args.unet_path else None
    
    # Process images
    results = batch_process_custom_method(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        prompts_dict=prompts_dict,
        unet_path=unet_path
    )
    
    print("\n✅ Custom two-stage method batch processing complete!")
