# save as: scripts/montage_confmat_vgg_vs_cyvgg.py
from PIL import Image, ImageDraw, ImageFont
import os

# --- PROVIDE YOUR OWN PATHS ---
# IMG_LEFT  = r"E:\...\confusion_matrix_VGG19-..._test_on_rotated-90-120.png"
# IMG_RIGHT = r"E:\...\confusion_matrix_CyVGG19-..._test_on_rotated-90-120.png"
IMG_LEFT  = r"E:\\MasterThesisLogsAll\\json_GTSRB\\confusion_matrices\\GTSRB-custom-vgg19-logpolar_dataset_GTSRB_non_rotated\\dataset_GTSRB_non_rotated_test_on_rotated-90-120\\confusion_matrix_row_norm.png"
IMG_RIGHT = r"E:\\MasterThesisLogsAll\\json_GTSRB\\confusion_matrices\\GTSRB-custom-cyvgg19-logpolar_dataset_GTSRB_non_rotated\\dataset_GTSRB_non_rotated_test_on_rotated-90-120\\confusion_matrix_row_norm.png"
TITLE_LEFT  = "VGG19-log — test: rotated-90-120"
TITLE_RIGHT = "CyVGG19-log — test: rotated-90-120"
OUT = r"D:\MasterThesis\MasterThesis\results\fig\confmat_VGG19_vs_CyVGG19_rot90-120.png"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

def load_resize_to_same_height(path, target_h=None):
    """Open an image and resize it to the given height (keeping aspect ratio)."""
    im = Image.open(path).convert("RGB")
    if target_h is None:
        return im
    w, h = im.size
    new_w = int(w * (target_h / h))
    return im.resize((new_w, target_h), Image.LANCZOS)

# 1) Load and align the two images by height
left0  = Image.open(IMG_LEFT).convert("RGB")
right0 = Image.open(IMG_RIGHT).convert("RGB")
H = min(left0.height, right0.height)
left  = load_resize_to_same_height(IMG_LEFT, H)
right = load_resize_to_same_height(IMG_RIGHT, H)

# 2) Title bar
pad = 16
title_h = 48
W = left.width + right.width + 3*pad
Htot = H + title_h + 2*pad

canvas = Image.new("RGB", (W, Htot), (255,255,255))
draw = ImageDraw.Draw(canvas)
# Use default font; if you prefer a TTF font:
# font = ImageFont.truetype("arial.ttf", 20)
font = ImageFont.load_default()

# 3) Draw titles
x_left = pad
x_right = pad + left.width + pad
y_title = pad
draw.text((x_left,  y_title), TITLE_LEFT,  fill=(0,0,0), font=font)
draw.text((x_right, y_title), TITLE_RIGHT, fill=(0,0,0), font=font)

# 4) Paste the confusion matrices
y_img = y_title + title_h - 8  # slight upward shift towards titles
canvas.paste(left,  (pad, y_img))
canvas.paste(right, (pad + left.width + pad, y_img))

# 5) Save the result
canvas.save(OUT, quality=95)
print(f"[OK] Saved: {OUT}")
