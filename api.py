import base64
import io
import shutil
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from main import (
	BASE_DIR,
	IMAGES_DIR,
	inpaint_background,
	prepare_images_dir,
	regenerate_vehicle,
	segment_edited_image,
	segment_image,
)

app = FastAPI(title="Vehicle Diff Edit API", version="1.0.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

FRONTEND_DIR = BASE_DIR / "frontend"
FAVICON_PATH = FRONTEND_DIR / "favicon.svg"

if FRONTEND_DIR.exists():
	app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

MODEL_FOLDER = BASE_DIR / "model"
UNET_PATH = MODEL_FOLDER / "unet_model_carvana_new.pth"
SD_MODEL_PATH = (
	MODEL_FOLDER
	/ "stable-diffusion"
 	/ "models--runwayml--stable-diffusion-v1-5"
	/ "snapshots"
	/ "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)


def ensure_model_files():
	if not UNET_PATH.exists():
		raise RuntimeError(f"Missing UNet weights at {UNET_PATH}")
	if not SD_MODEL_PATH.exists():
		raise RuntimeError(f"Missing Stable Diffusion model at {SD_MODEL_PATH}")


def ensure_run_dir(run_id: str) -> Path:
	run_dir = IMAGES_DIR / run_id
	run_dir.mkdir(parents=True, exist_ok=True)
	return run_dir


def encode_image(image_path: Path) -> str:
	if not image_path.exists():
		raise HTTPException(status_code=404, detail=f"Image not found: {image_path.name}")
	with Image.open(image_path) as img:
		buffer = io.BytesIO()
		img.save(buffer, format="PNG")
	return base64.b64encode(buffer.getvalue()).decode("utf-8")


def read_upload(upload: UploadFile, destination: Path) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	with destination.open("wb") as out_file:
		shutil.copyfileobj(upload.file, out_file)
	upload.file.close()


@app.on_event("startup")
async def on_startup():
	ensure_model_files()
	prepare_images_dir(IMAGES_DIR)


@app.get("/", response_class=HTMLResponse)
async def get_index():
	index_path = FRONTEND_DIR / "index.html"
	if not index_path.exists():
		raise HTTPException(status_code=404, detail="Frontend not found. Please build the UI.")
	return index_path.read_text(encoding="utf-8")


@app.get("/favicon.ico")
async def get_favicon():
	if FAVICON_PATH.exists():
		return FileResponse(FAVICON_PATH, media_type="image/svg+xml")
	raise HTTPException(status_code=404, detail="Favicon not found")


@app.post("/api/stage1")
async def stage1(file: UploadFile = File(...)):
	if file.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/webp"}:
		raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PNG, JPG, or WEBP image.")

	run_id = uuid4().hex
	run_dir = ensure_run_dir(run_id)
	original_path = run_dir / "original.png"

	read_upload(file, original_path)
	mask_path = Path(
		segment_image(str(original_path), str(UNET_PATH), output_dir=run_dir, output_name="stage1_mask.png")
	)

	return {
		"run_id": run_id,
		"stage": "stage1",
		"image": encode_image(mask_path),
		"label": "Stage 1 · Segmentation Mask",
	}


@app.post("/api/stage2")
async def stage2(
	run_id: str = Form(...),
	prompt: str = Form(...),
	negative_prompt: Optional[str] = Form(None),
):
	run_dir = ensure_run_dir(run_id)
	original_path = run_dir / "original.png"
	mask_path = run_dir / "stage1_mask.png"

	if not original_path.exists() or not mask_path.exists():
		raise HTTPException(status_code=400, detail="Stage 1 must be completed before Stage 2.")

	neg_prompt = negative_prompt or "blurry, low quality, distorted, ugly car, deformed vehicle"

	_, vehicle_path = regenerate_vehicle(
		str(original_path),
		str(mask_path),
		str(SD_MODEL_PATH),
		prompt=prompt,
		negative_prompt=neg_prompt,
		output_dir=run_dir,
		output_name="stage2_vehicle.png",
	)

	return {
		"run_id": run_id,
		"stage": "stage2",
		"image": encode_image(Path(vehicle_path)),
		"label": "Stage 2 · Vehicle Re-render",
	}


@app.post("/api/stage3")
async def stage3(run_id: str = Form(...)):
	run_dir = ensure_run_dir(run_id)
	edited_path = run_dir / "stage2_vehicle.png"

	if not edited_path.exists():
		raise HTTPException(status_code=400, detail="Stage 2 must be completed before Stage 3.")

	mask_path = segment_edited_image(
		str(edited_path),
		str(UNET_PATH),
		output_dir=run_dir,
		output_name="stage3_mask.png",
	)

	return {
		"run_id": run_id,
		"stage": "stage3",
		"image": encode_image(Path(mask_path)),
		"label": "Stage 3 · Updated Mask",
	}


@app.post("/api/stage4")
async def stage4(
	run_id: str = Form(...),
	prompt: str = Form(...),
	negative_prompt: Optional[str] = Form(None),
):
	run_dir = ensure_run_dir(run_id)
	edited_path = run_dir / "stage2_vehicle.png"
	mask_path = run_dir / "stage3_mask.png"

	if not edited_path.exists() or not mask_path.exists():
		raise HTTPException(status_code=400, detail="Stage 2 and Stage 3 must be completed before Stage 4.")

	neg_prompt = negative_prompt or "blurry, low quality, distorted, car, vehicle, text, watermark"

	_, final_path = inpaint_background(
		str(edited_path),
		str(mask_path),
		str(SD_MODEL_PATH),
		prompt=prompt,
		negative_prompt=neg_prompt,
		output_dir=run_dir,
		output_name="stage4_final.png",
	)

	return {
		"run_id": run_id,
		"stage": "stage4",
		"image": encode_image(Path(final_path)),
		"label": "Stage 4 · Final Composite",
	}