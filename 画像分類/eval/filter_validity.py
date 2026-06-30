"""
フィルタ（合成劣化）の妥当性を定量評価する。

問い: 自作の合成劣化が、本物 reCAPTCHA の劣化分布にどれだけ近いか。
近いほど「フィルタは妥当（本物に寄せられている）」と言える。

やること:
  クリーンな bus 画像（busbus/）に各フィルタを適用し、
  本物 reCAPTCHA bus 画像（real_recaptcha/bus/）との距離を測る。

  対象グループ:
    - clean    : 何もしない（劣化なし。基準）
    - old_rain : 旧フィルタ make_rainy_noise_image（暗く＋白い線）
    - old_night: 旧フィルタ make_night_image（暗く）
    - new      : 新フィルタ make_recaptcha_like_image（低解像度＋ぼけ＋JPEG）

  指標2つ:
    1. FID（Frechet Inception/Feature Distance）
       ResNet18 特徴で平均・共分散を取り、本物との距離を測る。小さいほど近い＝妥当。
    2. ドメイン分類器 AUC
       「本物 vs 合成」を見分けるロジスティック回帰の AUC（交差検証）。
       0.5 に近いほど見分けられない＝似ている＝妥当。

使い方:
  python 画像分類/eval/filter_validity.py
"""

import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torchvision import models, transforms

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_augment import (  # noqa: E402
    make_night_image,
    make_rainy_noise_image,
    make_recaptcha_like_image,
)

HERE = os.path.dirname(__file__)
CLEAN_BUS_DIR = os.path.join(HERE, "..", "..", "busbus")
REAL_BUS_DIR = os.path.join(HERE, "..", "real_recaptcha", "bus")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet 前処理（特徴抽出用）
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def load_feature_extractor():
    """ResNet18(ImageNet) の最終層手前 512 次元を出す特徴抽出器。"""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(DEVICE)
    return net


def load_images(paths):
    return [Image.open(p).convert("RGB") for p in paths]


@torch.no_grad()
def extract_features(net, pil_images):
    feats = []
    for img in pil_images:
        x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
        feats.append(net(x).cpu().numpy()[0])
    return np.stack(feats)


def fid(feat_a, feat_b):
    """2 つの特徴群の Frechet 距離。小さいほど分布が近い。"""
    mu_a, mu_b = feat_a.mean(0), feat_b.mean(0)
    cov_a = np.cov(feat_a, rowvar=False)
    cov_b = np.cov(feat_b, rowvar=False)
    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(cov_a @ cov_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov_a + cov_b - 2 * covmean))


def domain_auc(feat_synth, feat_real):
    """本物 vs 合成 を見分けるロジスティック回帰の AUC（5分割CV）。0.5 に近いほど似ている。"""
    X = np.vstack([feat_synth, feat_real])
    y = np.concatenate([np.zeros(len(feat_synth)), np.ones(len(feat_real))])
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def apply_filter(images, fn):
    return [fn(img.copy()) for img in images]


def main():
    clean_paths = sorted(
        glob.glob(os.path.join(CLEAN_BUS_DIR, "*.jpg"))
        + glob.glob(os.path.join(CLEAN_BUS_DIR, "*.png"))
    )
    real_paths = sorted(glob.glob(os.path.join(REAL_BUS_DIR, "*.png")))
    if not real_paths:
        raise SystemExit(
            f"本物画像が無い: {REAL_BUS_DIR}\n"
            "先に download_real_recaptcha.py を実行してください。"
        )

    print(f"clean bus: {len(clean_paths)} 枚 / real bus: {len(real_paths)} 枚")
    clean_imgs = load_images(clean_paths)
    real_imgs = load_images(real_paths)

    # 各グループの画像を用意（合成はクリーンに同じ枚数分フィルタ適用）
    groups = {
        "clean": clean_imgs,
        "old_night": apply_filter(clean_imgs, make_night_image),
        "old_rain": apply_filter(clean_imgs, make_rainy_noise_image),
        "new": apply_filter(clean_imgs, make_recaptcha_like_image),
    }

    net = load_feature_extractor()
    print("[feat] 本物の特徴抽出中...")
    real_feat = extract_features(net, real_imgs)

    rows = []
    for name, imgs in groups.items():
        print(f"[feat] {name} の特徴抽出中...")
        feat = extract_features(net, imgs)
        f = fid(feat, real_feat)
        auc = domain_auc(feat, real_feat)
        rows.append((name, f, auc))

    print("\n===== フィルタ妥当性（本物との近さ）=====")
    print(f"{'group':<10} {'FID(↓近い)':>14} {'domainAUC(0.5が理想)':>22}")
    for name, f, auc in rows:
        print(f"{name:<10} {f:>14.2f} {auc:>22.3f}")

    # 結果を CSV 保存
    out_csv = os.path.join(HERE, "results", "filter_validity.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as fp:
        fp.write("group,fid_to_real,domain_auc\n")
        for name, f, auc in rows:
            fp.write(f"{name},{f:.4f},{auc:.4f}\n")
    print(f"\n[saved] {out_csv}")

    # 結論の自動判定
    fid_map = {n: f for n, f, _ in rows}
    print("\n----- 読み方 -----")
    print(f"clean の FID={fid_map['clean']:.1f} を基準に、フィルタで本物に近づけば妥当。")
    for name in ("old_night", "old_rain", "new"):
        delta = fid_map["clean"] - fid_map[name]
        verdict = "近づいた(妥当)" if delta > 0 else "遠ざかった(逆効果)"
        print(f"  {name:<10}: FID {fid_map[name]:.1f}  (clean比 {delta:+.1f}) -> {verdict}")


if __name__ == "__main__":
    main()
