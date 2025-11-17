from pathlib import Path
from PIL import Image

# Folder containing your heatmaps
BASE_DIR = Path("/Users/maciej/PycharmProjects/MasterThesis/MasterThesis/media/assets/heatmaps")

# Output folder (compressed results go here)
OUT_BASE = BASE_DIR.parent / "heatmaps_compressed"

MAX_SIZE = 2000  # maximum width/height in pixels
JPG_QUALITY = 85  # 80–90 is typical, 85 is a good balance


def compress_image(png_path: Path):
    # Preserve directory structure
    rel = png_path.relative_to(BASE_DIR)
    out_dir = OUT_BASE / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(png_path).convert("RGB")
    w, h = img.size

    # Optionally resize if the image is too large
    scale = min(1.0, MAX_SIZE / max(w, h))
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    out_path = out_dir / (png_path.stem + ".jpg")

    img.save(
        out_path,
        format="JPEG",
        quality=JPG_QUALITY,
        optimize=True,
        progressive=True,
    )
    print(f"✔ {png_path}  →  {out_path}  (scale={scale:.2f})")


def main():
    png_files = list(BASE_DIR.rglob("*.png"))
    if not png_files:
        print(f"❌ No PNG files found under: {BASE_DIR}")
        return

    print(f"Found {len(png_files)} PNG files, compressing...")
    for p in png_files:
        compress_image(p)


if __name__ == "__main__":
    main()
