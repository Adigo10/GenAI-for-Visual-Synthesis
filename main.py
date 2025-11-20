import gc
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# For Stable Diffusion inpainting
from diffusers import StableDiffusionInpaintPipeline

# Directory where pipeline images are stored
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMAGE_SIZE = (256, 256)
MASK_THRESHOLD = 0.3    
SD_GUIDANCE_SCALE = 8
SD_INFERENCE_STEPS = 50

VEHICLE_PROMPT_SUFFIX = ""
BACKGROUND_PROMPT_SUFFIX = "high quality, realistic"

# Global pipeline for reuse
_global_pipeline = None
_global_device = None


def set_global_pipeline(pipeline, device):
    """Set the global pipeline for reuse."""
    global _global_pipeline, _global_device
    _global_pipeline = pipeline
    _global_device = device


def _resolve_output_dir(path: Optional[Path] = None) -> Path:
    output_dir = Path(path) if path else IMAGES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_and_resize_image(img_path: str | Path, mode: str = "RGB") -> Image.Image:
    return Image.open(img_path).convert(mode).resize(IMAGE_SIZE)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = (np.array(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(array).unsqueeze(0)


def _stitch_prompt(user_prompt: Optional[str], suffix: str) -> str:
    """Append quality suffix to user prompt while avoiding duplicate commas."""
    cleaned_prompt = (user_prompt or "").strip(" ,")
    if cleaned_prompt and suffix:
        return f"{cleaned_prompt}, {suffix}"
    return cleaned_prompt or suffix


def _save_mask(mask_array: np.ndarray, output_dir: Path, output_name: str) -> str:
    mask_img = (mask_array > MASK_THRESHOLD).astype("uint8") * 255
    mask_pil = Image.fromarray(mask_img).convert("L")
    mask_output = output_dir / output_name
    mask_pil.save(mask_output)
    return str(mask_output)


def _clear_torch_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _load_unet(model_path, device=None) -> "UNet":
    """Load UNet model with GPU support.
    
    Args:
        model_path: Path to UNet model
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    unet = UNet()
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    unet.load_state_dict(state_dict)
    unet.eval()
    unet = unet.to(device)
    return unet


@contextmanager
def load_inpaint_pipeline(model_dir=None, device=None, use_global=True):
    """Load Stable Diffusion inpainting pipeline with GPU support.
    
    Args:
        model_dir: Path to model directory (optional, ignored if use_global=True and global pipeline exists)
        device: Device to use ('cuda', 'cpu', 'mps', or None for auto-detect)
        use_global: If True, use the global pipeline if available
    """
    global _global_pipeline, _global_device
    
    # Use global pipeline if available
    if use_global and _global_pipeline is not None:
        try:
            yield _global_pipeline, _global_device
            return
        except Exception:
            pass
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("🍎 Using Apple Silicon MPS")
        else:
            try:
                import torch_directml
                device = torch_directml.device()
                print("Using DirectML (AMD GPU)")
            except Exception:
                device = "cpu"
                print("⚠️  Using CPU (this will be slow!)")
    
    # Use float16 for GPU, float32 for CPU
    dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
    
    # Load from HuggingFace or local path
    if model_dir and Path(model_dir).exists():
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=dtype,
        )
    
    pipe = pipe.to(device)

    try:
        yield pipe, device
    finally:
        if not use_global:
            del pipe
            _clear_torch_cache()


def load_unified_prompts(prompts_file: Optional[Path] = None) -> dict:
    """Load unified prompts from JSON file.
    
    Args:
        prompts_file: Path to unified_prompts.json file
        
    Returns:
        Dictionary mapping image names to prompts
    """
    if prompts_file is None:
        prompts_file = OUTPUTS_DIR / "unified_prompts.json"
    
    if not prompts_file.exists():
        print(f"⚠️  Warning: Prompt file not found: {prompts_file}")
        print("   Using default prompts. Run prompt generation cell first for better results.")
        return {}
    
    with open(prompts_file, 'r') as f:
        prompts = json.load(f)
    
    print(f"✓ Loaded prompts for {len(prompts)} images from {prompts_file}")
    return prompts


def prepare_images_dir(images_dir: Optional[Path] = None):
    """Create images folder if missing. If it contains files, clear them.

    This ensures each run starts with an empty `images/` directory.
    """
    if images_dir is None:
        images_dir = IMAGES_DIR

    images_dir = Path(images_dir)

    if not images_dir.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created images directory: {images_dir}")
        return

    # If folder exists and is not empty, clear its contents
    entries = list(images_dir.iterdir())
    if entries:
        print(f"Images directory not empty ({len(entries)} items). Clearing...")
        for path in entries:
            try:
                if path.is_file() or path.is_symlink():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: failed to remove {path}: {e}")
        print("Images directory cleared")
    else:
        print("Images directory exists and is empty")

# --- Step 1: Foreground-Background Segmentation (U-Net) ---
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Output layer
        self.out = torch.nn.Conv2d(64, out_channels, 1)
        
        self.pool = torch.nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_ch, out_ch):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output with sigmoid activation
        out = torch.sigmoid(self.out(d1))
        return out


def _segment_and_save_mask(
    img_path,
    model_path,
    output_dir: Optional[Path],
    output_name: str,
    device=None,
):
    """Segment image and save mask with GPU support.
    
    Args:
        img_path: Path to input image
        model_path: Path to UNet model
        output_dir: Output directory for mask
        output_name: Output filename for mask
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = _resolve_output_dir(output_dir)
    unet = _load_unet(model_path, device=device)

    image = _load_and_resize_image(img_path)
    img_tensor = _image_to_tensor(image).to(device)

    with torch.inference_mode():
        mask = unet(img_tensor).squeeze().cpu().numpy()

    mask_path = _save_mask(mask, output_dir, output_name)

    del unet
    _clear_torch_cache()
    return mask_path

def segment_image(img_path, model_path, output_dir: Optional[Path] = None, output_name: str = "stage1_mask.png", device=None):
    """Segment image to create mask (Stage 1).
    
    This identifies the vehicle region using UNet segmentation.
    
    Args:
        img_path: Path to input image
        model_path: Path to UNet model
        output_dir: Output directory (default: IMAGES_DIR)
        output_name: Output filename
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    return _segment_and_save_mask(img_path, model_path, output_dir, output_name, device=device)

# --- Step 2: Vehicle Regeneration (Stable Diffusion Inpainting) ---
def regenerate_vehicle(img_path, mask_path, model_dir=None, prompt=None, target_prompt=None, source_prompt=None, negative_prompt="blurry, low quality, distorted, ugly car, deformed vehicle", output_dir: Optional[Path] = None, output_name: str = "stage2_vehicle.png", device=None):
    """
    Use Stable Diffusion to regenerate the vehicle (Stage 2).
    
    This transforms the source vehicle into the target vehicle using inpainting.
    White in mask = vehicle to regenerate
    Black in mask = background to keep
    
    Args:
        img_path: Path to input image
        mask_path: Path to mask from Stage 1
        model_dir: Path to Stable Diffusion model (optional, uses global pipeline if available)
        prompt: Vehicle prompt (simple form)
        target_prompt: Target subject description (what you want) - for unified prompts
        source_prompt: Source subject description (what you have) - optional, for reference
        negative_prompt: What to avoid in generation
        output_dir: Output directory
        output_name: Output filename
        device: Device to use ('cuda', 'cpu', 'mps', or None for auto-detect)
    """
    # Use target_prompt if provided (unified prompts), otherwise use prompt
    vehicle_prompt = target_prompt if target_prompt else prompt
    full_prompt = _stitch_prompt(vehicle_prompt, VEHICLE_PROMPT_SUFFIX)
    
    # Print transformation info
    if source_prompt:
        print(f"   Transforming: {source_prompt}")
        print(f"   → Target: {vehicle_prompt}")

    with load_inpaint_pipeline(model_dir, device=device, use_global=True) as (pipe, _):
        image = _load_and_resize_image(img_path)
        mask = _load_and_resize_image(mask_path, mode="L")

        # Use mask directly: white = areas to regenerate (vehicle), black = keep (background)
        result_img = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=SD_INFERENCE_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE,
        ).images[0]

    output_dir = _resolve_output_dir(output_dir)
    vehicle_output = output_dir / output_name
    result_img.save(vehicle_output)

    return result_img, str(vehicle_output)

# --- Step 3: Re-segment the edited image ---
def segment_edited_image(img_path, model_path, output_dir: Optional[Path] = None, output_name="stage3_mask.png", device=None):
    """
    Perform segmentation again on the edited vehicle image (Stage 3).
    This creates a fresh mask for the newly generated vehicle.
    
    Args:
        img_path: Path to edited vehicle image from Stage 2
        model_path: Path to UNet model
        output_dir: Output directory (default: IMAGES_DIR)
        output_name: Output filename
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
    """
    output_path = _segment_and_save_mask(img_path, model_path, output_dir, output_name, device=device)
    return output_path

# --- Step 4: Background Inpainting (Stable Diffusion) ---
def inpaint_background(img_path, mask_path, model_dir=None, prompt: Optional[str] = None, negative_prompt="blurry, low quality, distorted, washed out, duplicate, text, watermark, jpeg artifacts, vehicles, cars", output_dir: Optional[Path] = None, output_name: str = "stage4_final.png", device=None):
    """
    Use Stable Diffusion inpainting to fill in the background (Stage 4).
    Black in mask = background to inpaint
    White in mask = foreground to keep
    
    Args:
        img_path: Path to edited vehicle image from Stage 2
        mask_path: Path to mask from Stage 3
        model_dir: Path to Stable Diffusion model (optional, uses global pipeline if available)
        prompt: Background description
        negative_prompt: What to avoid in generation
        output_dir: Output directory
        output_name: Output filename
        device: Device to use ('cuda', 'cpu', 'mps', or None for auto-detect)
    """
    full_prompt = _stitch_prompt(prompt, BACKGROUND_PROMPT_SUFFIX)

    with load_inpaint_pipeline(model_dir, device=device, use_global=True) as (pipe, _):
        image = _load_and_resize_image(img_path)
        mask = _load_and_resize_image(mask_path, mode="L")

        # Invert mask: white = keep (vehicle), black = inpaint (background)
        mask_array = np.array(mask)
        inverted_mask = 255 - mask_array
        mask_inverted = Image.fromarray(inverted_mask).convert("L")

        result_img = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_inverted,
            num_inference_steps=SD_INFERENCE_STEPS,
            guidance_scale=SD_GUIDANCE_SCALE
        ).images[0]

    output_dir = _resolve_output_dir(output_dir)
    final_output = output_dir / output_name
    result_img.save(final_output)

    return result_img, str(final_output)

# --- Entire Workflow (Four-Stage Custom Pipeline) ---
if __name__ == "__main__":
    # Configuration
    img_path = "0a2bbd5330a2_03.jpg"
    img_name = Path(img_path).name
    model_folder = BASE_DIR / "model"
    
    # Path to UNet model for segmentation
    unet_path = model_folder / "unet_model_carvana_new.pth"
    
    # Path to Stable Diffusion model
    sd_model_path = (
        model_folder
        / "stable-diffusion"
        / "models--runwayml--stable-diffusion-inpainting"
        / "snapshots"
        / "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )

    print("Starting Four-Stage Custom Pipeline...")
    print("="*60)
    
    # Load unified prompts
    unified_prompts = load_unified_prompts()
    
    # Get prompts for this image
    if img_name in unified_prompts:
        prompts = unified_prompts[img_name]
        source_subject = prompts.get("source_subject", "a car")
        target_subject = prompts.get("target_subject", "a sleek modern sports car")
        background_prompt = prompts.get("background", "sunlit alpine highway")
        original_caption = prompts.get("original_caption", "")
        
        print(f"\n📝 Using prompts for {img_name}:")
        print(f"   Original: {original_caption}")
        print(f"   Source: {source_subject}")
        print(f"   Target: {target_subject}")
        print(f"   Background: {background_prompt}")
    else:
        print(f"\n⚠️  No prompts found for {img_name}, using defaults")
        source_subject = "a car"
        target_subject = "a sleek modern sports car"
        background_prompt = "sunlit alpine highway"
    
    # Ensure images directory exists and is cleared
    prepare_images_dir(IMAGES_DIR)
    
    print("\n" + "="*60)
    print("STAGE 1: Initial Vehicle Segmentation (UNet)")
    print("="*60)
    print("Purpose: Identify vehicle region in original image")
    print(f"Input: {img_path}")
    mask_path = segment_image(img_path, unet_path)
    print(f"✓ Initial mask saved to: {mask_path}")

    print("\n" + "="*60)
    print("STAGE 2: Vehicle Regeneration (Stable Diffusion)")
    print("="*60)
    print("Purpose: Transform source vehicle → target vehicle")
    print(f"Input: {img_path} + {mask_path}")
    vehicle_img, vehicle_path = regenerate_vehicle(
        img_path, 
        mask_path, 
        sd_model_path, 
        target_prompt=target_subject,
        source_prompt=source_subject
    )
    print(f"✓ Edited vehicle saved to: {vehicle_path}")

    print("\n" + "="*60)
    print("STAGE 3: Re-segmentation of Edited Vehicle (UNet)")
    print("="*60)
    print("Purpose: Create fresh mask for the newly generated vehicle")
    print(f"Input: {vehicle_path}")
    new_mask_path = segment_edited_image(vehicle_path, unet_path, output_name="stage3_mask.png")
    print(f"✓ Re-segmentation mask saved to: {new_mask_path}")

    print("\n" + "="*60)
    print("STAGE 4: Background Inpainting (Stable Diffusion)")
    print("="*60)
    print("Purpose: Generate new background around the edited vehicle")
    print(f"Input: {vehicle_path} + {new_mask_path}")
    print(f"Prompt: {background_prompt}")
    final_img, final_path = inpaint_background(
        vehicle_path, 
        new_mask_path, 
        sd_model_path, 
        prompt=background_prompt
    )
    print(f"✓ Final image with inpainted background saved to: {final_path}")

    print("\n" + "="*60)
    print("🎉 Four-Stage Pipeline Complete!")
    print("="*60)
    print("\nPipeline Summary:")
    print(f"  Stage 1 (Initial Segmentation): {IMAGES_DIR / 'stage1_mask.png'}")
    print(f"  Stage 2 (Vehicle Regeneration): {IMAGES_DIR / 'stage2_vehicle.png'}")
    print(f"  Stage 3 (Re-segmentation): {IMAGES_DIR / 'stage3_mask.png'}")
    print(f"  Stage 4 (Background Inpainting): {IMAGES_DIR / 'stage4_final.png'}")
    print(f"\nTransformation:")
    print(f"  Vehicle: {source_subject} → {target_subject}")
    print(f"  Background: {background_prompt}")