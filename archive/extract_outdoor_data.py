import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

# extract_outdoor_data.py
# GitHub Copilot
#
# Download ADE20K (ADEChallengeData2016), find images whose scene labels
# indicate "outdoor" and copy them into a sorted output folder.
#
# Usage:
#   python extract_outdoor_data.py --outdir ./ade_outdoor --keep-zip
#
# Notes:
# - The ADE20K archive is downloaded from MIT CSAIL's public mirror.
# - The script searches for JSON/text metadata files that contain scene
#   labels. It looks for common keys and a list of outdoor-related keywords.
# - If metadata can't be found/parsed automatically, the script will list
#   what it found and exit with instructions.


try:
    TQDM = tqdm
except Exception:
    TQDM = lambda x, **kw: x  # fallback


DATASET_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
DEFAULT_ZIP_NAME = "ADEChallengeData2016.zip"

# Keywords indicating outdoor scenes (case-insensitive substring match)
OUTDOOR_KEYWORDS = {
    "outdoor",
    "exterior",
    "street",
    "highway",
    "field",
    "beach",
    "mountain",
    "forest",
    "park",
    "plaza",
    "square",
    "desert",
    "valley",
    "river",
    "sea",
    "shore",
    "lake",
    "garden",
    "yard",
    "sky",
    "stadium",
    "playground",
    "bridge",
    "harbor",
    "bay",
    "cliff",
    "waterfront",
}


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with a streaming request (uses urllib to avoid heavy deps)."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r:
        total = r.length or None
        with open(dest, "wb") as f:
            if total:
                for chunk in TQDM(iter(lambda: r.read(chunk_size), b""), total=total // chunk_size + 1, unit="chunk"):
                    f.write(chunk)
            else:
                while True:
                    chunk = r.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)


def unzip_once(zip_path: Path, extract_to: Path) -> None:
    print(f"Unzipping {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def find_image_dirs(root: Path) -> List[Path]:
    candidates = []
    for name in ["images", "ADEChallengeData2016", "images/training", "images/validation"]:
        p = root / name
        if p.exists() and p.is_dir():
            candidates.append(p)
    # fallback: any directory with many jpg/png files
    if not candidates:
        for sub in root.rglob("*"):
            if sub.is_dir():
                cnt = sum(1 for _ in sub.glob("*.jpg")) + sum(1 for _ in sub.glob("*.png"))
                if cnt >= 50:
                    candidates.append(sub)
    # dedupe and return
    unique = []
    seen = set()
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            unique.append(c)
            seen.add(s)
    return unique


def text_contains_outdoor(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in OUTDOOR_KEYWORDS)


def parse_possible_metadata_file(path: Path) -> Dict[str, str]:
    """
    Try to parse a metadata file and return a mapping filename -> scene_label.
    Supports several plausible formats:
    - JSON arrays/dicts with per-image entries (look for keys like 'file_name','image','image_path','scene','scene_label')
    - Plain text files mapping "image_name scene_name" or "image_name\tscene"
    """
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    # quick plain-text mapping attempt (lines with two columns)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and len(lines) >= 5:
        # detect lines like: "ADE_train_00000001.jpg\tstreet"
        simple_pairs = []
        for ln in lines:
            if "\t" in ln:
                a, b = ln.split("\t", 1)
                simple_pairs.append((a.strip(), b.strip()))
            elif " " in ln:
                a, b = ln.split(maxsplit=1)
                simple_pairs.append((a.strip(), b.strip()))
        if simple_pairs:
            for a, b in simple_pairs:
                mapping[a] = b
            if mapping:
                return mapping

    # try JSON
    try:
        obj = json.loads(text)
    except Exception:
        return mapping

    # If obj is a dict with 'images' or 'annotations' keys
    candidates = []
    if isinstance(obj, dict):
        for key in ("images", "annotations", "metadata", "labels"):
            if key in obj and isinstance(obj[key], list):
                candidates.append(obj[key])
        # also if top-level dict maps filenames to scene names
        if all(isinstance(v, str) for v in obj.values()):
            for k, v in obj.items():
                mapping[k] = v
            if mapping:
                return mapping
    elif isinstance(obj, list):
        candidates.append(obj)

    for arr in candidates:
        for entry in arr:
            if not isinstance(entry, dict):
                continue
            # possible filename keys
            fname = None
            for fk in ("file_name", "filename", "image", "img_name", "image_name", "imagePath", "path"):
                if fk in entry:
                    fname = entry[fk]
                    break
            # possible scene keys
            scene = None
            for sk in ("scene", "scene_name", "scene_class", "sceneLabel", "label", "place"):
                if sk in entry:
                    scene = entry[sk]
                    break
            # sometimes 'annotation' contains nested info
            if scene is None and "annotation" in entry and isinstance(entry["annotation"], dict):
                for sk in ("scene", "scene_name"):
                    if sk in entry["annotation"]:
                        scene = entry["annotation"][sk]
                        break
            if fname and scene and isinstance(fname, str) and isinstance(scene, str):
                mapping[Path(fname).name] = scene
    return mapping


def collect_metadata_mappings(root: Path) -> Dict[str, str]:
    """
    Search for metadata files and aggregate mappings. Returns mapping image_basename -> scene_label.
    """
    mappings: Dict[str, str] = {}
    # prioritize files/folders that include 'scene' or 'place' in name
    for p in root.rglob("*"):
        if p.is_file():
            low = p.name.lower()
            if any(k in low for k in ("scene", "place", "metadata", "imageInfo", "img_info", "annotations", "labels")):
                try:
                    m = parse_possible_metadata_file(p)
                    if m:
                        mappings.update(m)
                        print(f"Loaded {len(m)} mappings from {p}")
                except Exception as e:
                    print(f"Skipping {p} (parse error): {e}")
    return mappings


def copy_outdoor_images(mappings: Dict[str, str], image_dirs: Iterable[Path], outdir: Path) -> Tuple[int, int]:
    """
    Copy images whose scene label matches OUTDOOR_KEYWORDS.
    Returns (copied_count, total_matched).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    copied = 0
    matched = 0
    # build lookup of available images: basename -> fullpath (first found)
    image_lookup: Dict[str, Path] = {}
    for d in image_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in d.glob(ext):
                name = p.name
                if name not in image_lookup:
                    image_lookup[name] = p
    for fname, scene in mappings.items():
        if not isinstance(scene, str):
            continue
        if text_contains_outdoor(scene):
            matched += 1
            src = image_lookup.get(Path(fname).name)
            if src and src.exists():
                dst = outdir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
                    copied += 1
            else:
                # try searching by prefix or partial match
                for name, p in image_lookup.items():
                    if Path(fname).stem == Path(name).stem:
                        dst = outdir / p.name
                        if not dst.exists():
                            shutil.copy2(p, dst)
                            copied += 1
                            break
    return copied, matched


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Download ADE20K and extract images with outdoor scenes.")
    p.add_argument("--outdir", type=Path, default=Path("ade_outdoor"), help="Output directory to store outdoor images")
    p.add_argument("--tmpdir", type=Path, default=Path("ade_tmp"), help="Temp directory for download/unzip")
    p.add_argument("--zip", type=Path, default=Path(DEFAULT_ZIP_NAME), help="Path to ADE zip (will download if absent)")
    p.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip in place")
    args = p.parse_args(argv)

    zip_path = args.zip
    tmpdir = args.tmpdir
    outdir = args.outdir

    if not zip_path.exists():
        try:
            download_file(DATASET_URL, zip_path)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

    # unzip
    unzip_dir = tmpdir / "ADEChallengeData2016"
    if not unzip_dir.exists():
        unzip_once(zip_path, tmpdir)

    # find image directories
    image_dirs = find_image_dirs(tmpdir)
    if not image_dirs:
        print("No image directories found after unzipping. Contents:")
        for p in tmpdir.iterdir():
            print("  ", p)
        sys.exit(1)
    print("Found image directories:")
    for d in image_dirs:
        print("  -", d)

    # collect metadata mappings
    mappings = collect_metadata_mappings(tmpdir)
    if not mappings:
        print("No metadata mappings found automatically. Please inspect the unzipped folder and provide a mapping file.")
        print("Candidate metadata files (look for 'imageInfo' or 'scene' files) under:", tmpdir)
        sys.exit(1)

    # copy outdoor images
    copied, matched = copy_outdoor_images(mappings, image_dirs, outdir)
    print(f"Matched {matched} images labeled as outdoor; copied {copied} files to {outdir}")

    if not args.keep_zip:
        try:
            zip_path.unlink(missing_ok=True)
        except Exception:
            pass
    print("Done.")


if __name__ == "__main__":
    main()