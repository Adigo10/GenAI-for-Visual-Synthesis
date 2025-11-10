import torch
import numpy as np
from PIL import Image
import gc
import os

# For Stable Diffusion inpainting
from diffusers import StableDiffusionInpaintPipeline

# --- Step 1: Foreground-Background Segmentation (U-Net) ---
class UNet(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder (upsampling)
        self.upconv4 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        
        self.upconv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        
        self.upconv2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        
        self.upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # Output layer
        self.out = torch.nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    
    def _block(self, in_channels, out_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Output
        return torch.sigmoid(self.out(dec1))

def segment_image(img_path, model_path):
    # Load model
    unet = UNet()
    unet.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    unet.eval()

    image = Image.open(img_path).convert("RGB").resize((256, 256))
    img_tensor = torch.tensor(
        (np.array(image) / 255.0).transpose(2, 0, 1),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        mask = unet(img_tensor).squeeze().numpy()
    mask_img = (mask > 0.5).astype("uint8") * 255
    mask_pil = Image.fromarray(mask_img).convert("L")
    mask_pil.save("stage1_mask.png")

    del unet
    torch.cuda.empty_cache()
    gc.collect()
    return "stage1_mask.png"

# --- Step 2: Background Inpainting (Stable Diffusion) ---
def inpaint_background(img_path, mask_path, model_dir):
    """
    Use Stable Diffusion inpainting to fill in the background.
    The mask should be inverted so that white areas are inpainted.
    """
    # Load Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # Use DirectML for AMD GPUs on Windows, or CPU
    try:
        import torch_directml
        device = torch_directml.device()
        pipe = pipe.to(device)
        print("Using DirectML (AMD GPU)")
    except:
        device = "cpu"
        pipe = pipe.to(device)
        print("Using CPU")

    # Load image and mask
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Invert mask: white = areas to inpaint (background), black = keep (foreground)
    mask_array = np.array(mask)
    inverted_mask = 255 - mask_array
    mask_inverted = Image.fromarray(inverted_mask).convert("L")

    # Generate inpainted background
    prompt = "natural outdoor background, high quality, photorealistic"
    negative_prompt = "blurry, low quality, distorted"
    
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_inverted,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    
    result_img.save("stage2_inpainted.png")

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return "stage2_inpainted.png"

# --- Step 3: Semantic Editing (Stable Diffusion Inpainting) ---
def semantic_edit(img_path, mask_path, model_dir, prompt):
    """
    Use Stable Diffusion to perform semantic editing on specific regions.
    This can refine or change specific parts of the image based on the prompt.
    """
    # Load Stable Diffusion Inpainting Pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    
    # Use DirectML for AMD GPUs on Windows, or CPU
    try:
        import torch_directml
        device = torch_directml.device()
        pipe = pipe.to(device)
        print("Using DirectML (AMD GPU)")
    except:
        device = "cpu"
        pipe = pipe.to(device)
        print("Using CPU")

    # Load image and mask
    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # For semantic editing, we might want to edit the background
    # Invert mask: white = areas to edit (background), black = keep (foreground)
    mask_array = np.array(mask)
    inverted_mask = 255 - mask_array
    mask_inverted = Image.fromarray(inverted_mask).convert("L")

    # Generate edited image with custom prompt
    negative_prompt = "blurry, low quality, distorted, ugly"
    
    result_img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask_inverted,
        num_inference_steps=50,
        guidance_scale=7.5,
        strength=0.8  # Controls how much to change the image
    ).images[0]
    
    result_img.save("stage3_final.png")

    del pipe
    if device != "cpu":
        torch.cuda.empty_cache()
    gc.collect()
    return "stage3_final.png"

# --- Entire Workflow ---
if __name__ == "__main__":
    img_path = "0a0e3fb8f782_01.jpg"
    model_folder = "model"
    
    # Path to UNet model for segmentation
    unet_path = os.path.join(model_folder, "unet_model.pth")
    
    # Path to Stable Diffusion model
    sd_model_path = os.path.join(
        model_folder, 
        "stable-diffusion",
        "models--runwayml--stable-diffusion-v1-5",
        "snapshots",
        "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
    )

    print("Starting image editing pipeline...")
    
    # Stage 1: Segmentation
    print("\n[Stage 1] Segmenting foreground/background...")
    mask_path = segment_image(img_path, unet_path)
    print(f"✓ Mask saved to: {mask_path}")

    # Stage 2: Inpainting
    print("\n[Stage 2] Inpainting background with Stable Diffusion...")
    inpainted_path = inpaint_background(img_path, mask_path, sd_model_path)
    print(f"✓ Inpainted image saved to: {inpainted_path}")

    # Stage 3: Semantic Editing
    print("\n[Stage 3] Applying semantic editing with Stable Diffusion...")
    custom_prompt = "sleek modern sedan car, beautiful sunset beach background, golden hour, photorealistic, high quality, luxury sedan"
    final_path = semantic_edit(inpainted_path, mask_path, sd_model_path, prompt=custom_prompt)
    print(f"✓ Final edited image saved to: {final_path}")

    print(f"\n🎉 Pipeline complete! Final image: {final_path}")
