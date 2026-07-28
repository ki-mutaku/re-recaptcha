"""
判定しきい値を「val で決めて、test に適用する」正しい手順で評価する。

これまでの compare_models.py の maxF1 は、評価データ自身でしきい値を探した値
（テストでチューニング＝わずかにリーク）だった。ここでは
  1. dataset/val（学習に使っていない検証データ）でF1が最大になるしきい値を探す
  2. そのしきい値を固定して 本物reCAPTCHA（test）に適用する
という順にして、実運用に即した正直な数字を出す。

比較用に、しきい値0.5（素朴な既定値）での結果も並べて表示する。

使い方（リポジトリのルートで）:
  uv run python 画像分類/eval/tune_threshold.py
  uv run python 画像分類/eval/tune_threshold.py --model best_resnet18_bus_BASELINE.pth
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
VAL_DIR = os.path.join(REPO, "dataset", "val")
REAL_DIR = os.path.join(HERE, "..", "real_recaptcha")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageList(Dataset):
    """(path, label) のリストを読み込むだけの単純なDataset。"""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return preprocess(img), label


def collect(root, pos_dir, neg_dir, exts):
    items = []
    for name, label in ((pos_dir, 1), (neg_dir, 0)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(exts):
                items.append((os.path.join(d, f), label))
    return items


def score_all(model, bus_idx, items, device, batch_size=64):
    """全画像の「バスっぽさ」確率を返す。"""
    loader = DataLoader(ImageList(items), batch_size=batch_size, num_workers=0)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, bus_idx]
            scores.extend(p.cpu().tolist())
            labels.extend(y.tolist())
    return scores, labels


def prf(scores, labels, th):
    tp = sum(1 for s, y in zip(scores, labels) if s >= th and y == 1)
    fp = sum(1 for s, y in zip(scores, labels) if s >= th and y == 0)
    fn = sum(1 for s, y in zip(scores, labels) if s < th and y == 1)
    tn = sum(1 for s, y in zip(scores, labels) if s < th and y == 0)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(labels) if labels else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="best_resnet18_bus.pth", help="評価するモデル(.pth)")
    args = ap.parse_args()

    model_path = os.path.join(REPO, args.model)
    classes_path = os.path.join(REPO, "best_resnet18_bus_classes.json")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    class_names = json.load(open(classes_path))
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    bus_idx = class_names.index("bus")

    print(f"モデル: {args.model}  デバイス: {device}\n")

    # --- 1. val でしきい値を決める ---
    val_items = collect(VAL_DIR, "bus", "other", (".jpg", ".jpeg", ".png"))
    if not val_items:
        raise SystemExit(f"検証データがありません: {VAL_DIR}")
    print(f"[val] {len(val_items)}枚 でしきい値を探索中...")
    v_scores, v_labels = score_all(model, bus_idx, val_items, device)

    best_th, best_f1 = 0.5, -1.0
    for i in range(1, 100):
        th = i / 100
        f1 = prf(v_scores, v_labels, th)["f1"]
        if f1 > best_f1:
            best_f1, best_th = f1, th
    vm = prf(v_scores, v_labels, best_th)
    print(f"  → val最適しきい値 = {best_th:.2f} (val F1={vm['f1']:.3f})\n")

    # --- 2. そのしきい値を本物reCAPTCHA(test)に適用 ---
    test_items = collect(REAL_DIR, "bus", "nonbus", (".png", ".jpg"))
    if not test_items:
        raise SystemExit(f"本物データがありません: {REAL_DIR}")
    print(f"[test] 本物reCAPTCHA {len(test_items)}枚 を判定中...")
    t_scores, t_labels = score_all(model, bus_idx, test_items, device)

    print("\n===== 本物reCAPTCHA での結果 =====")
    print(f"{'しきい値':<22}{'正解率':<10}{'適合率':<10}{'再現率':<10}{'F1'}")
    print("-" * 62)
    for label, th in ((f"0.50 (素朴な既定値)", 0.5), (f"{best_th:.2f} (valで決定)", best_th)):
        m = prf(t_scores, t_labels, th)
        print(f"{label:<22}{m['acc']:<10.3f}{m['precision']:<10.3f}"
              f"{m['recall']:<10.3f}{m['f1']:.3f}")

    print("\n※ valで決めたしきい値をtestに適用＝テストでチューニングしていない正直な数字。")


if __name__ == "__main__":
    main()
