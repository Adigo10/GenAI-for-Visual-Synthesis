import gc
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

IMAGE_SIZE = (256, 256)
MASK_THRESHOLD = 0.3    
SD_GUIDANCE_SCALE = 10
SD_INFERENCE_STEPS = 50

VEHICLE_PROMPT_SUFFIX = ""
BACKGROUND_PROMPT_SUFFIX = "high quality, realistic"


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


def _load_unet(model_path) -> "UNet":
    unet = UNet()
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location="cpu")
    unet.load_state_dict(state_dict)
    unet.eval()
    return unet


@contextmanager
def load_inpaint_pipeline(model_dir):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    )

    try:
        import torch_directml
        device = torch_directml.device()
        pipe = pipe.to(device)
        print("Using DirectML (AMD GPU)")
    except Exception:
        device = "cpu"
        pipe = pipe.to(device)
        print("Using CPU")

    try:
        yield pipe, device
    finally:
        del pipe
        if device != "cpu":
            _clear_torch_cache()
        else:
            gc.collect()


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
):
    output_dir = _resolve_output_dir(output_dir)
    unet = _load_unet(model_path)

    image = _load_and_resize_image(img_path)
    img_tensor = _image_to_tensor(image)

    with torch.inference_mode():
        mask = unet(img_tensor).squeeze().numpy()

    mask_path = _save_mask(mask, output_dir, output_name)

    del unet
    _clear_torch_cache()
    return mask_path

def segment_image(img_path, model_path, output_dir: Optional[Path] = None, output_name: str = "stage1_mask.png"):
    return _segment_and_save_mask(img_path, model_path, output_dir, output_name)

# --- Step 2: Vehicle Regeneration (Stable Diffusion Inpainting) ---
def regenerate_vehicle(img_path, mask_path, model_dir, prompt, negative_prompt="blurry, low quality, distorted, ugly car, deformed vehicle", output_dir: Optional[Path] = None, output_name: str = "stage2_vehicle.png"):
    """
    Use Stable Diffusion to regenerate the vehicle (white zone of mask).
    White in mask = vehicle to regenerate
    Black in mask = background to keep
    """
    full_prompt = _stitch_prompt(prompt, VEHICLE_PROMPT_SUFFIX)

    with load_inpaint_pipeline(model_dir) as (pipe, _):
        image = _load_and_resize_image(img_path)
        mask = _load_and_resize_image(mask_path, mode="L")

        # Use mask directly: white = areas to regenerate (vehicle), black = keep (background)
        result_img = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            mask_threshold=MASK_THRESHOLD,
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
def segment_edited_image(img_path, model_path, output_dir: Optional[Path] = None, output_name="stage3_mask.png"):
    """
    Perform segmentation again on the edited vehicle image.
    This creates a fresh mask for the newly generated vehicle.
    """
    output_path = _segment_and_save_mask(img_path, model_path, output_dir, output_name)
    print(f"✓ Re-segmentation mask saved to: {output_path}")
    return output_path

# --- Step 4: Background Inpainting (Stable Diffusion) ---
def inpaint_background(img_path, mask_path, model_dir, prompt: Optional[str] = None, negative_prompt="blurry, low quality, distorted, washed out, duplicate, text, watermark, jpeg artifacts, vehicles, cars", output_dir: Optional[Path] = None, output_name: str = "stage4_final.png"):
    """
    Use Stable Diffusion inpainting to fill in the background (black zone of mask).
    Black in mask = background to inpaint
    White in mask = foreground to keep
    """
    full_prompt = _stitch_prompt(prompt, BACKGROUND_PROMPT_SUFFIX)

    with load_inpaint_pipeline(model_dir) as (pipe, _):
        image = _load_and_resize_image(img_path)
        mask = _load_and_resize_image(mask_path, mode="L")

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

    print(f"✓ Final image with inpainted background saved to: {final_output}")
    return result_img, str(final_output)

# --- Entire Workflow ---
if __name__ == "__main__":
    img_path = "0a2bbd5330a2_03.jpg"
    model_folder = BASE_DIR / "model"
    
    # Path to UNet model for segmentation
    unet_path = model_folder / "unet_model_carvana_new.pth"
    
    # Path to Stable Diffusion model
    sd_model_path = (
        model_folder
        / "stable-diffusion"
        / "models--runwayml--stable-diffusion-v1-5"
        / "snapshots"
        / "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )

    print("Starting image editing pipeline...")
    print("="*60)
    # Ensure images directory exists and is cleared if needed
    prepare_images_dir(IMAGES_DIR)
    
    # Stage 1: Initial Segmentation (UNet/SAM)
    print("\n[Stage 1] Initial segmentation with UNet...")
    print("Purpose: Identify vehicle region in original image")
    mask_path = segment_image(img_path, unet_path)
    print(f"✓ Initial mask saved to: {mask_path}")

    # Stage 2: Vehicle Regeneration/Editing
    print("\n[Stage 2] Regenerating vehicle with Stable Diffusion...")
    print("Purpose: Generate edits for the car/vehicle")
    vehicle_prompt_input = "sleek sports coupe"  # Replace with user-provided prompt text
    vehicle_img, vehicle_path = regenerate_vehicle(img_path, mask_path, sd_model_path, prompt=vehicle_prompt_input)
    print(f"✓ Edited vehicle image saved to: {vehicle_path}")

    # Stage 3: Re-segmentation on edited image
    print("\n[Stage 3] Re-segmenting the edited vehicle image with UNet...")
    print("Purpose: Create fresh mask for the newly generated vehicle")
    new_mask_path = segment_edited_image(vehicle_path, unet_path, output_name="stage3_mask.png")

    # Stage 4: Final Background Inpainting
    print("\n[Stage 4] Inpainting background with Stable Diffusion...")
    print("Purpose: Generate new background around the edited vehicle")
    background_prompt_input = "sunlit alpine highway"  # Replace with user-provided background prompt
    final_img, final_path = inpaint_background(vehicle_path, new_mask_path, sd_model_path, prompt=background_prompt_input)

    print("\n" + "="*60)
    print(f"🎉 Pipeline complete! Final image: {final_path}")
    print("\nPipeline summary:")
    print(f"  1. Initial segmentation → {IMAGES_DIR / 'stage1_mask.png'}")
    print(f"  2. Vehicle editing → {IMAGES_DIR / 'stage2_vehicle.png'}")
    print(f"  3. Re-segmentation → {IMAGES_DIR / 'stage3_mask.png'}")
    print(f"  4. Background inpainting → {IMAGES_DIR / 'stage4_final.png'}")

