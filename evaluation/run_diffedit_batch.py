"""
Batch DiffEdit Processing Script
=================================
This script runs DiffEdit on a batch of test images for evaluation purposes.
Based on the DiffEdit notebook implementation.
"""

import torch
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import json


def setup_diffedit_pipeline(device="cuda"):
    """Initialize DiffEdit pipeline with proper schedulers"""
    
    print("Loading DiffEdit pipeline...")
    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True,
    )
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    
    # Optimizations
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    
    print("✓ DiffEdit pipeline loaded!")
    return pipeline


def process_image_with_diffedit(
    pipeline,
    image_path: Path,
    source_prompt: str,
    target_prompt: str,
    output_dir: Path,
    save_mask: bool = True,
    image_size: tuple = (768, 768)
):
    """
    Process a single image with DiffEdit
    
    Args:
        pipeline: DiffEdit pipeline
        image_path: Path to input image
        source_prompt: Description of what to change (e.g., "a black SUV")
        target_prompt: Description of desired result (e.g., "a black Sedan")
        output_dir: Directory to save outputs
        save_mask: Whether to save the generated mask
        image_size: Size to resize images to
    
    Returns:
        Tuple of (output_image, mask_image, output_path, mask_path)
    """
    
    # Load and resize image
    raw_image = load_image(str(image_path)).resize(image_size)
    
    # Generate mask
    mask_image = pipeline.generate_mask(
        image=raw_image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    
    # Invert latents
    inv_latents = pipeline.invert(
        prompt=source_prompt,
        image=raw_image
    ).latents
    
    # Generate final image
    output_image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        negative_prompt=source_prompt,
    ).images[0]
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    img_name = image_path.name
    output_path = output_dir / f"edited_{img_name}"
    output_image.save(output_path)
    
    mask_path = None
    if save_mask:
        # Convert mask to PIL Image
        mask_pil = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L")
        mask_pil = mask_pil.resize(image_size)
        mask_path = output_dir / f"mask_{img_name}"
        mask_pil.save(mask_path)
    
    return output_image, mask_image, output_path, mask_path


def batch_process_diffedit(
    input_dir: Path,
    output_dir: Path,
    prompts_dict: dict,
    device: str = "cuda",
    save_masks: bool = True,
    image_size: tuple = (768, 768)
):
    """
    Process multiple images with DiffEdit
    
    Args:
        input_dir: Directory with input images
        output_dir: Directory to save outputs
        prompts_dict: Dictionary with source and target prompts per image
            Format: {
                "image1.jpg": {
                    "source": "a black SUV",
                    "target": "a red sports car"
                },
                ...
            }
        device: Device to use
        save_masks: Whether to save generated masks
        image_size: Size to resize images to
    """
    
    # Setup pipeline
    pipeline = setup_diffedit_pipeline(device=device)
    
    # Get list of images
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([
        f for f in input_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    print(f"\nProcessing {len(image_files)} images with DiffEdit...")
    print(f"{'='*60}\n")
    
    results = []
    
    for img_path in tqdm(image_files, desc="Processing images"):
        img_name = img_path.name
        
        # Get prompts for this image
        if img_name not in prompts_dict:
            print(f"⚠ No prompts found for {img_name}, skipping...")
            continue
        
        prompts = prompts_dict[img_name]
        source_prompt = prompts.get('source', '')
        target_prompt = prompts.get('target', '')
        
        if not source_prompt or not target_prompt:
            print(f"⚠ Missing source or target prompt for {img_name}, skipping...")
            continue
        
        try:
            # Process image
            output_img, mask, output_path, mask_path = process_image_with_diffedit(
                pipeline=pipeline,
                image_path=img_path,
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                output_dir=output_dir,
                save_mask=save_masks,
                image_size=image_size
            )
            
            results.append({
                'input': str(img_path),
                'output': str(output_path),
                'mask': str(mask_path) if mask_path else None,
                'source_prompt': source_prompt,
                'target_prompt': target_prompt,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Error processing {img_name}: {e}")
            results.append({
                'input': str(img_path),
                'output': None,
                'mask': None,
                'source_prompt': source_prompt,
                'target_prompt': target_prompt,
                'status': 'error',
                'error': str(e)
            })
    
    # Save processing log
    log_path = output_dir / "processing_log.json"
    with open(log_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processing complete!")
    print(f"✓ Processed {len([r for r in results if r['status'] == 'success'])} images successfully")
    print(f"✓ Log saved to: {log_path}")
    
    return results


def load_prompts_from_file(prompts_file: Path) -> dict:
    """
    Load prompts from JSON file
    
    Expected format:
    {
        "image1.jpg": {
            "source": "a black SUV",
            "target": "a red sports car"
        },
        "image2.jpg": {
            "source": "original description",
            "target": "desired description"
        }
    }
    """
    with open(prompts_file, 'r') as f:
        return json.load(f)


def create_sample_prompts_file(output_path: Path):
    """Create a sample prompts file for reference"""
    sample_prompts = {
        "example1.jpg": {
            "source": "a black SUV",
            "target": "a red sports car"
        },
        "example2.jpg": {
            "source": "a sedan on a road",
            "target": "a convertible on a beach"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(sample_prompts, f, indent=2)
    
    print(f"✓ Sample prompts file created: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DiffEdit on batch of images")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save DiffEdit outputs")
    parser.add_argument("--prompts", type=str, required=True,
                       help="JSON file with source/target prompts per image")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--save_masks", action="store_true",
                       help="Save generated masks")
    parser.add_argument("--image_size", type=int, nargs=2, default=[768, 768],
                       help="Image size (width height)")
    parser.add_argument("--create_sample_prompts", action="store_true",
                       help="Create a sample prompts file and exit")
    
    args = parser.parse_args()
    
    # Create sample prompts file if requested
    if args.create_sample_prompts:
        create_sample_prompts_file(Path("sample_prompts.json"))
        exit(0)
    
    # Load prompts
    prompts_dict = load_prompts_from_file(Path(args.prompts))
    print(f"✓ Loaded prompts for {len(prompts_dict)} images")
    
    # Process images
    results = batch_process_diffedit(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        prompts_dict=prompts_dict,
        device=args.device,
        save_masks=args.save_masks,
        image_size=tuple(args.image_size)
    )
    
    print("\n✅ DiffEdit batch processing complete!")
