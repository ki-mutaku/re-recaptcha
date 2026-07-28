"""
ブートストラップ信頼区間計算スクリプト (bootstrap_ci.py)

- 目的: zero-shot ResNet と FT後ResNet の AP 差分（ΔAP = FT - ZS）について、
        統計的なばらつき（評価セットに対する不確実性）を測るため、
        ペアド・ブートストラップ信頼区間（95%CI、10,000回リサンプル）を計算する。
- 対象データセット:
    - 晴れ (img)
    - 雨 (img_bus_rain)
    - 本物 (real_recaptcha)
- 出力:
    - src/image_classification/eval/results/bootstrap_ci.md (表)
    - src/image_classification/eval/results/bootstrap_ci.csv
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageOps

# 必要なパスのインポート
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification import (
    build_preprocess,
    collect_image_paths,
    load_resnet_classifier,
    target_confidence_score,
)
from evaluate import (
    load_labels,
    match_labels_to_paths,
)
from compare_models import (
    build_ft_preprocess,
    load_ft_model,
    FT_MODEL_PATH,
    FT_CLASSES_PATH,
    DATASETS,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
N_BOOTSTRAPS = 10000
SEED = 42

def compute_ap_fast(scores, labels):
    """APを高速に計算する（ベクトル化版）"""
    sort_idx = np.argsort(scores)[::-1]
    sorted_labels = labels[sort_idx]
    
    total_pos = np.sum(labels)
    if total_pos == 0:
        return 0.0
        
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(~sorted_labels)
    precision = tp / (tp + fp)
    
    ap = np.sum(precision * sorted_labels) / total_pos
    return ap

def get_scores_and_labels(image_dir, labels_path, device, target="bus"):
    """指定データセットの画像について、モデル推論を行いスコアとラベルを配列で返す"""
    image_paths = collect_image_paths(image_dir)
    label_dict = load_labels(labels_path)
    labeled_data = match_labels_to_paths(image_paths, label_dict)
    
    if not labeled_data:
        raise ValueError(f"No labeled images found in {image_dir} matching {labels_path}")
        
    # ラベル抽出
    labels = np.array([is_pos for _, is_pos in labeled_data], dtype=bool)
    
    # 1. Zero-shot スコアの計算
    zs_model, categories = load_resnet_classifier()
    zs_model = zs_model.to(device)
    zs_model.eval()
    zs_preprocess = build_preprocess()
    
    zs_scores = []
    # 2. FT スコアの計算
    ft_model, class_names = load_ft_model(FT_MODEL_PATH, FT_CLASSES_PATH)
    ft_model = ft_model.to(device)
    ft_model.eval()
    ft_preprocess = build_ft_preprocess()
    bus_idx = class_names.index(target)
    
    ft_scores = []
    
    total = len(labeled_data)
    print(f"Inference on {total} images from {Path(image_dir).name} using {device}...")
    
    # 1枚ずつ推論（GPUが使える場合は十分高速）
    for count, (idx, _) in enumerate(labeled_data):
        path = image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Zero-shot
        padded_image = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
        input_tensor_zs = zs_preprocess(padded_image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_zs = torch.softmax(zs_model(input_tensor_zs)[0], dim=0).cpu()
        zs_score = target_confidence_score(prob_zs, categories, target)
        zs_scores.append(zs_score)
        
        # FT
        input_tensor_ft = ft_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_ft = torch.softmax(ft_model(input_tensor_ft)[0], dim=0).cpu()
        ft_score = prob_ft[bus_idx].item()
        ft_scores.append(ft_score)
        
        if (count + 1) % 2000 == 0 or (count + 1) == total:
            print(f"  Processed {count + 1}/{total} images...")
            
    return np.array(zs_scores), np.array(ft_scores), labels

def run_bootstrap(zs_scores, ft_scores, labels, n_bootstraps=N_BOOTSTRAPS, seed=SEED):
    """ペアド・ブートストラップを実行し、AP値の配列を返す"""
    rng = np.random.default_rng(seed)
    n = len(labels)
    
    zs_aps = np.zeros(n_bootstraps)
    ft_aps = np.zeros(n_bootstraps)
    diff_aps = np.zeros(n_bootstraps)
    
    print(f"Running {n_bootstraps} bootstraps...")
    for i in range(n_bootstraps):
        # ペアでのサンプリング（同じ画像インデックスのセットを使用）
        indices = rng.choice(n, size=n, replace=True)
        
        zs_ap = compute_ap_fast(zs_scores[indices], labels[indices])
        ft_ap = compute_ap_fast(ft_scores[indices], labels[indices])
        
        zs_aps[i] = zs_ap
        ft_aps[i] = ft_ap
        diff_aps[i] = ft_ap - zs_ap
        
        if (i + 1) % 2000 == 0:
            print(f"  Completed {i + 1}/{n_bootstraps} resamples...")
            
    return zs_aps, ft_aps, diff_aps

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not FT_MODEL_PATH.exists():
        print(f"Error: FT model not found at {FT_MODEL_PATH}. Run train_resnet.py first.")
        sys.exit(1)
        
    results = []
    
    for image_dir, labels_path, display in DATASETS:
        if not Path(image_dir).exists():
            print(f"Skip: {image_dir} does not exist.")
            continue
            
        print(f"\n=== Evaluating {display} ===")
        zs_scores, ft_scores, labels = get_scores_and_labels(image_dir, labels_path, device)
        
        # オリジナルのAP
        zs_orig = compute_ap_fast(zs_scores, labels)
        ft_orig = compute_ap_fast(ft_scores, labels)
        diff_orig = ft_orig - zs_orig
        
        # ブートストラップ
        zs_aps, ft_aps, diff_aps = run_bootstrap(zs_scores, ft_scores, labels)
        
        # 95%信頼区間の計算 (2.5% から 97.5% パーセンタイル)
        zs_ci = np.percentile(zs_aps, [2.5, 97.5])
        ft_ci = np.percentile(ft_aps, [2.5, 97.5])
        diff_ci = np.percentile(diff_aps, [2.5, 97.5])
        
        # 0を跨ぐかの判定
        crosses_zero = (diff_ci[0] <= 0 <= diff_ci[1])
        crosses_zero_str = "跨ぐ" if crosses_zero else "跨がない"
        
        print(f"  ZS AP: {zs_orig:.3f} (95% CI: [{zs_ci[0]:.3f}, {zs_ci[1]:.3f}])")
        print(f"  FT AP: {ft_orig:.3f} (95% CI: [{ft_ci[0]:.3f}, {ft_ci[1]:.3f}])")
        print(f"  ΔAP  : {diff_orig:+.3f} (95% CI: [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}]) - 0を{crosses_zero_str}")
        
        results.append({
            "dataset": display,
            "zs_ap": zs_orig,
            "zs_ci_low": zs_ci[0],
            "zs_ci_high": zs_ci[1],
            "ft_ap": ft_orig,
            "ft_ci_low": ft_ci[0],
            "ft_ci_high": ft_ci[1],
            "diff_ap": diff_orig,
            "diff_ci_low": diff_ci[0],
            "diff_ci_high": diff_ci[1],
            "crosses_zero": crosses_zero_str
        })
        
    # 保存処理
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. CSV 保存
    csv_path = DEFAULT_OUTPUT_DIR / "bootstrap_ci.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved CSV results to: {csv_path}")
    
    # 2. Markdown 保存
    md_path = DEFAULT_OUTPUT_DIR / "bootstrap_ci.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AP差分のブートストラップ信頼区間評価 (95% CI)\n\n")
        f.write("リサンプル回数: 10,000回、乱数シード: 42（ペアド・ブートストラップ法）\n\n")
        f.write("| データセット | zero-shot AP (95% CI) | fine-tuned AP (95% CI) | 差分 ΔAP (95% CI) | 0を跨ぐか |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            zs_str = f"{r['zs_ap']:.3f} ([{r['zs_ci_low']:.3f}, {r['zs_ci_high']:.3f}])"
            ft_str = f"{r['ft_ap']:.3f} ([{r['ft_ci_low']:.3f}, {r['ft_ci_high']:.3f}])"
            diff_str = f"{r['diff_ap']:+.3f} ([{r['diff_ci_low']:.3f}, {r['diff_ci_high']:.3f}])"
            f.write(f"| {r['dataset']} | {zs_str} | {ft_str} | {diff_str} | {r['crosses_zero']} |\n")
    print(f"Saved Markdown report to: {md_path}")

if __name__ == "__main__":
    main()
