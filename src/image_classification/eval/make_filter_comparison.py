"""
フィルタ比較画像を生成する。

同じバス画像に clean / old_night / old_rain / new の4フィルタを適用し、
横並びの比較画像として保存する。

使い方:
  python src/image_classification/eval/make_filter_comparison.py
  python src/image_classification/eval/make_filter_comparison.py --n 4  # 4パターン保存
"""

import argparse
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from data_augment import make_night_image, make_rainy_noise_image, make_recaptcha_like_image

REAL_BUS_DIR = os.path.join(HERE, "..", "data", "real_recaptcha", "bus")
OUT_DIR = os.path.join(HERE, "results")

FILTERS = [
    ("clean\n（劣化なし）",    lambda img: img.copy()),
    ("old_night\n（暗く）",   make_night_image),
    ("old_rain\n（暗く＋白線）", make_rainy_noise_image),
    ("new\n（低解像度＋ぼけ＋JPEG）", make_recaptcha_like_image),
]

CELL = 160      # 1セルのpxサイズ（表示用にリサイズ）
LABEL_H = 44    # ラベル行の高さ
PAD = 12        # セル間の余白
MARGIN = 20     # 外側の余白
BG = (245, 245, 242)
BORDER = (200, 200, 195)


def draw_label(draw, x, y, w, h, text):
    """ラベルを中央揃えで描く（2行対応）"""
    lines = text.split("\n")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    total_h = len(lines) * 16
    start_y = y + (h - total_h) // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (w - tw) // 2, start_y + i * 16), line, fill=(50, 50, 50), font=font)


def make_comparison(source_path, out_path):
    n = len(FILTERS)
    total_w = MARGIN * 2 + n * CELL + (n - 1) * PAD
    total_h = MARGIN * 2 + LABEL_H + CELL

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw = ImageDraw.Draw(canvas)

    random.seed(42)
    with Image.open(source_path).convert("RGB") as src:
        for i, (label, fn) in enumerate(FILTERS):
            random.seed(42)
            filtered = fn(src.copy()).resize((CELL, CELL), Image.BILINEAR)

            x = MARGIN + i * (CELL + PAD)
            y_label = MARGIN
            y_img = MARGIN + LABEL_H

            draw_label(draw, x, y_label, CELL, LABEL_H, label)
            canvas.paste(filtered, (x, y_img))
            draw.rectangle([x, y_img, x + CELL - 1, y_img + CELL - 1], outline=BORDER)

    canvas.save(out_path)
    print(f"[saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="保存するサンプル数")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    pngs = sorted(f for f in os.listdir(REAL_BUS_DIR) if f.endswith(".png"))
    if not pngs:
        raise SystemExit(f"本物画像が無い: {REAL_BUS_DIR}")

    targets = pngs[5:5 + args.n]
    for i, fname in enumerate(targets):
        src = os.path.join(REAL_BUS_DIR, fname)
        out = os.path.join(OUT_DIR, f"filter_comparison_{i+1:02d}.png")
        make_comparison(src, out)


if __name__ == "__main__":
    main()
