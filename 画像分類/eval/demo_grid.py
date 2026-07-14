"""
本物reCAPTCHAタイルで 3×3 グリッドを組み、学習済みFTモデルに解かせるデモ。

real_recaptcha/{bus,nonbus}/ から bus と nonbus のタイル（各100×100・本物）を
ランダムに選んで 3×3 に並べ、reCAPTCHA 風のグリッド画像を作る。
各マスを FTモデル（best_resnet18_bus.pth）で「バスか否か」判定し、
モデルが選んだマスと正解を並べて表示・画像保存する。

タイルは自分で選ぶので各マスの正解が既知 → 「何マス正解したか」まで言える。

使い方（リポジトリのルートで）:
  uv run python 画像分類/eval/demo_grid.py                 # ランダムに1枚デモ
  uv run python 画像分類/eval/demo_grid.py --buses 4       # バスを4マスに
  uv run python 画像分類/eval/demo_grid.py --seed 7        # 並びを固定（再現・発表用）
  uv run python 画像分類/eval/demo_grid.py --threshold 0.5 # 判定しきい値

出力:
  eval/results/demo_grid_input.png   … 出題（3×3グリッド）
  eval/results/demo_grid_result.png  … 判定結果（緑枠=モデルがバスと判定 / 赤枠=間違い）
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MODEL_PATH = os.path.join(REPO, "best_resnet18_bus.pth")
CLASSES_PATH = os.path.join(REPO, "best_resnet18_bus_classes.json")
BUS_DIR = os.path.join(HERE, "..", "real_recaptcha", "bus")
NONBUS_DIR = os.path.join(HERE, "..", "real_recaptcha", "nonbus")
OUT_DIR = os.path.join(HERE, "results")

TILE = 100      # 1マスのpxサイズ（本物が100×100）
GAP = 6         # マス間の隙間
BG = (255, 255, 255)
GREEN = (34, 170, 60)
RED = (210, 40, 40)

# 学習時(train_resnet.py val)と同じ前処理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    class_names = json.load(open(CLASSES_PATH))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, class_names.index("bus")


def bus_prob(model, bus_idx, pil_img):
    x = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.softmax(model(x)[0], dim=0)
    return prob[bus_idx].item()


def pick_tiles(n_bus, seed):
    random.seed(seed)
    bus_files = sorted(f for f in os.listdir(BUS_DIR) if f.endswith(".png"))
    non_files = sorted(f for f in os.listdir(NONBUS_DIR) if f.endswith(".png"))
    chosen = [(os.path.join(BUS_DIR, f), True) for f in random.sample(bus_files, n_bus)] + \
             [(os.path.join(NONBUS_DIR, f), False) for f in random.sample(non_files, 9 - n_bus)]
    random.shuffle(chosen)  # バスの位置をばらす
    return chosen  # [(path, is_bus_truth)] 長さ9


def compose_grid(tiles):
    size = TILE * 3 + GAP * 4
    canvas = Image.new("RGB", (size, size), BG)
    imgs = []
    for i, (path, _) in enumerate(tiles):
        im = Image.open(path).convert("RGB").resize((TILE, TILE))
        r, c = divmod(i, 3)
        x = GAP + c * (TILE + GAP)
        y = GAP + r * (TILE + GAP)
        canvas.paste(im, (x, y))
        imgs.append(im)
    return canvas, imgs


def annotate(tiles, preds, truths):
    """判定結果を可視化。緑枠=モデルがバスと判定。赤枠=間違い（誤選択 or 見逃し）。"""
    size = TILE * 3 + GAP * 4
    canvas = Image.new("RGB", (size, size), BG)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    for i, (path, truth) in enumerate(tiles):
        im = Image.open(path).convert("RGB").resize((TILE, TILE))
        r, c = divmod(i, 3)
        x = GAP + c * (TILE + GAP)
        y = GAP + r * (TILE + GAP)
        canvas.paste(im, (x, y))
        d = ImageDraw.Draw(canvas)
        pred = preds[i]
        correct = (pred == truth)
        # 枠: モデルがバス判定=緑、間違いは赤で上書き
        if pred:  # モデルがバスと判定 → 選択
            color = GREEN if correct else RED
            for w in range(4):
                d.rectangle([x - w, y - w, x + TILE + w, y + TILE + w], outline=color)
            d.text((x + 4, y + 2), "BUS", fill=color, font=font)
        elif truth and not pred:  # 見逃し（バスなのに選ばなかった）
            for w in range(4):
                d.rectangle([x - w, y - w, x + TILE + w, y + TILE + w], outline=RED)
            d.text((x + 4, y + 2), "miss", fill=RED, font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buses", type=int, default=3, help="バスを何マスにするか(1-8)")
    ap.add_argument("--seed", type=int, default=random.randint(0, 9999), help="並びの乱数シード")
    ap.add_argument("--threshold", type=float, default=0.5, help="バス判定のしきい値")
    args = ap.parse_args()
    args.buses = max(1, min(8, args.buses))

    os.makedirs(OUT_DIR, exist_ok=True)
    model, bus_idx = load_model()
    tiles = pick_tiles(args.buses, args.seed)

    grid_img, tile_imgs = compose_grid(tiles)
    in_path = os.path.join(OUT_DIR, "demo_grid_input.png")
    grid_img.save(in_path)

    # 各マスを判定
    preds, probs, truths = [], [], []
    for (path, truth), im in zip(tiles, tile_imgs):
        p = bus_prob(model, bus_idx, im)
        probs.append(p)
        preds.append(p >= args.threshold)
        truths.append(truth)

    res_img = annotate(tiles, preds, truths)
    res_path = os.path.join(OUT_DIR, "demo_grid_result.png")
    res_img.save(res_path)

    # コンソール表示（3×3）
    print(f"\n=== 3×3 reCAPTCHA デモ (seed={args.seed}, threshold={args.threshold}) ===")
    print("お題: バスが写っているマスを全て選べ\n")
    print("  正解グリッド（B=バス）        モデルの選択（O=選ぶ）      確信度")
    for r in range(3):
        truth_row = "  ".join("B" if truths[r*3+c] else "・" for c in range(3))
        pred_row = "  ".join("O" if preds[r*3+c] else "・" for c in range(3))
        prob_row = " ".join(f"{probs[r*3+c]:.2f}" for c in range(3))
        print(f"   {truth_row}          {pred_row}         {prob_row}")

    n_correct = sum(p == t for p, t in zip(preds, truths))
    print(f"\n9マス中 {n_correct} マス正解")
    if all(p == t for p, t in zip(preds, truths)):
        print("→ 全マス正解！ このグリッドは突破できる。")
    print(f"\n出題画像: {in_path}")
    print(f"結果画像: {res_path}")


if __name__ == "__main__":
    main()
