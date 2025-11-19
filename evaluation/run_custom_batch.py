"""
Batch Processing Script for Custom Method
==========================================
Runs the custom UNet + Stable Diffusion pipeline on multiple test images.
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm import tqdm
import shutil

# Import custom pipeline functions
from main import (
    segment_image,
    regenerate_vehicle,
    segment_edited_image,
    inpaint_background,
    BASE_DIR
)


def process_image_with_custom_method(
    img_path: Path,
    unet_path: Path,
    sd_model_path: Path,
    vehicle_prompt: str,
    background_prompt: str,
    output_dir: Path,
    vehicle_negative_prompt: str = "blurry, low quality, distorted, ugly car, deformed vehicle",
    background_negative_prompt: str = "blurry, low quality, distorted, washed out, duplicate, text, watermark, jpeg artifacts, vehicles, cars"
):
    """
    Process a single image through the custom 4-stage pipeline
    
    Args:
        img_path: Path to input image
        unet_path: Path to UNet segmentation model
        sd_model_path: Path to Stable Diffusion model
        vehicle_prompt: Prompt for vehicle regeneration
        background_prompt: Prompt for background inpainting
        output_dir: Directory to save all outputs
        vehicle_negative_prompt: Negative prompt for vehicle generation
        background_negative_prompt: Negative prompt for background generation
    
    Returns:
        Dictionary with paths to all generated outputs
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name = img_path.name
    results = {
        'input': str(img_path),
        'stage1_mask': None,
        'stage2_vehicle': None,
        'stage3_mask': None,
        'stage4_final': None
    }
    
    try:
        # Stage 1: Initial Segmentation
        stage1_mask_name = f"stage1_mask_{img_name}"
        stage1_mask_path = segment_image(
            img_path=str(img_path),
            model_path=str(unet_path),
            output_dir=output_dir,
            output_name=stage1_mask_name
        )
        results['stage1_mask'] = stage1_mask_path
        
        # Stage 2: Vehicle Regeneration
        stage2_vehicle_name = f"stage2_vehicle_{img_name}"
        _, stage2_vehicle_path = regenerate_vehicle(
            img_path=str(img_path),
            mask_path=stage1_mask_path,
            model_dir=str(sd_model_path),
            prompt=vehicle_prompt,
            negative_prompt=vehicle_negative_prompt,
            output_dir=output_dir,
            output_name=stage2_vehicle_name
        )
        results['stage2_vehicle'] = stage2_vehicle_path
        
        # Stage 3: Re-segmentation
        stage3_mask_name = f"stage3_mask_{img_name}"
        stage3_mask_path = segment_edited_image(
            img_path=stage2_vehicle_path,
            model_path=str(unet_path),
            output_dir=output_dir,
            output_name=stage3_mask_name
        )
        results['stage3_mask'] = stage3_mask_path
        
        # Stage 4: Background Inpainting
        stage4_final_name = f"final_{img_name}"
        _, stage4_final_path = inpaint_background(
            img_path=stage2_vehicle_path,
            mask_path=stage3_mask_path,
            model_dir=str(sd_model_path),
            prompt=background_prompt,
            negative_prompt=background_negative_prompt,
            output_dir=output_dir,
            output_name=stage4_final_name
        )
        results['stage4_final'] = stage4_final_path
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
    unet_path: Optional[Path] = None,
    sd_model_path: Optional[Path] = None
):
    """
    Process multiple images with custom method
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save all outputs
        prompts_dict: Dictionary with prompts per image
            Format: {
                "image1.jpg": {
                    "vehicle": "red sports car",
                    "background": "mountain road"
                },
                ...
            }
        unet_path: Path to UNet model (defaults to model/unet_model_carvana_new.pth)
        sd_model_path: Path to SD model (defaults to model/stable-diffusion/...)
    """
    
    # Set default model paths if not provided
    if unet_path is None:
        unet_path = BASE_DIR / "model" / "unet_model_carvana_new.pth"
    
    if sd_model_path is None:
        sd_model_path = (
            BASE_DIR / "model" / "stable-diffusion" / 
            "models--runwayml--stable-diffusion-v1-5" / "snapshots" /
            "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
        )
    
    # Validate model paths
    if not unet_path.exists():
        raise FileNotFoundError(f"UNet model not found: {unet_path}")
    if not sd_model_path.exists():
        raise FileNotFoundError(f"Stable Diffusion model not found: {sd_model_path}")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    print(f"\nProcessing {len(image_files)} images with Custom Method...")
    print(f"{'='*60}\n")
    print(f"UNet model: {unet_path}")
    print(f"SD model: {sd_model_path}")
    print(f"Output directory: {output_dir}\n")
    
    all_results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = img_path.name
        
        # Get prompts for this image
        if img_name not in prompts_dict:
            print(f"⚠ No prompts found for {img_name}, skipping...")
            continue
        
        prompts = prompts_dict[img_name]
        vehicle_prompt = prompts.get('vehicle', '')
        background_prompt = prompts.get('background', '')
        
        if not vehicle_prompt or not background_prompt:
            print(f"⚠ Missing vehicle or background prompt for {img_name}, skipping...")
            continue
        
        # Process image
        results = process_image_with_custom_method(
            img_path=img_path,
            unet_path=unet_path,
            sd_model_path=sd_model_path,
            vehicle_prompt=vehicle_prompt,
            background_prompt=background_prompt,
            output_dir=output_dir
        )
        
        results['vehicle_prompt'] = vehicle_prompt
        results['background_prompt'] = background_prompt
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
    - final_images/ - Stage 4 final outputs
    - masks/ - Stage 1 and Stage 3 masks
    - intermediate/ - Stage 2 vehicle images
    """
    
    final_dir = output_dir / "final_images"
    masks_dir = output_dir / "masks"
    intermediate_dir = output_dir / "intermediate"
    
    final_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    intermediate_dir.mkdir(exist_ok=True)
    
    for result in results:
        if result.get('status') != 'success':
            continue
        
        # Copy final image
        if result['stage4_final']:
            src = Path(result['stage4_final'])
            dst = final_dir / src.name.replace('final_', '')
            if src.exists():
                shutil.copy2(src, dst)
        
        # Copy stage 3 mask (best mask for evaluation)
        if result['stage3_mask']:
            src = Path(result['stage3_mask'])
            dst = masks_dir / src.name.replace('stage3_mask_', '')
            if src.exists():
                shutil.copy2(src, dst)
        
        # Copy intermediate vehicle image
        if result['stage2_vehicle']:
            src = Path(result['stage2_vehicle'])
            dst = intermediate_dir / src.name
            if src.exists():
                shutil.copy2(src, dst)
    
    print(f"\n✓ Outputs organized:")
    print(f"  - Final images: {final_dir}")
    print(f"  - Masks: {masks_dir}")
    print(f"  - Intermediate: {intermediate_dir}")


def load_prompts_from_file(prompts_file: Path) -> dict:
    """
    Load prompts from JSON file
    
    Expected format:
    {
        "image1.jpg": {
            "vehicle": "red sports car",
            "background": "mountain road at sunset"
        },
        "image2.jpg": {
            "vehicle": "blue sedan",
            "background": "city street"
        }
    }
    """
    with open(prompts_file, 'r') as f:
        return json.load(f)


def create_sample_prompts_file(output_path: Path):
    """Create a sample prompts file for reference"""
    sample_prompts = {
        "example1.jpg": {
            "vehicle": "sleek red sports car",
            "background": "winding mountain road at sunset"
        },
        "example2.jpg": {
            "vehicle": "blue luxury sedan",
            "background": "modern city street with skyscrapers"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_prompts, f, indent=2)
    
    print(f"✓ Sample prompts file created: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Custom Method on batch of images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save custom method outputs")
    parser.add_argument("--prompts", type=str, required=True,
                       help="JSON file with vehicle/background prompts per image")
    parser.add_argument("--unet_path", type=str, default=None,
                       help="Path to UNet model (optional)")
    parser.add_argument("--sd_model_path", type=str, default=None,
                       help="Path to Stable Diffusion model (optional)")
    parser.add_argument("--create_sample_prompts", action="store_true",
                       help="Create a sample prompts file and exit")
    
    args = parser.parse_args()
    
    # Create sample prompts file if requested
    if args.create_sample_prompts:
        create_sample_prompts_file(Path("sample_prompts_custom.json"))
        exit(0)
    
    # Load prompts
    prompts_dict = load_prompts_from_file(Path(args.prompts))
    print(f"✓ Loaded prompts for {len(prompts_dict)} images")
    
    # Parse optional model paths
    unet_path = Path(args.unet_path) if args.unet_path else None
    sd_model_path = Path(args.sd_model_path) if args.sd_model_path else None
    
    # Process images
    results = batch_process_custom_method(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        prompts_dict=prompts_dict,
        unet_path=unet_path,
        sd_model_path=sd_model_path
    )
    
    print("\n✅ Custom method batch processing complete!")
