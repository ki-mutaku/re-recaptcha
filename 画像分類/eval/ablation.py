"""
アブレーション実験スクリプト (ablation.py)

- 役割: 学習用データの劣化条件を変えた3つの条件でモデルを学習し、
        同じ real_recaptcha で AP を比較して、新フィルタの有効性を因果として検証する。
- 条件:
  - A: clean (原本のみ、100枚/クラス) -> 新規学習 (best_resnet18_bus_ablation_clean.pth)
  - B: old_filter (原本＋旧夜＋旧雨、300枚/クラス) -> 新規学習 (best_resnet18_bus_ablation_oldfilter.pth)
  - C: new_filter (原本＋新degraded1＋新degraded2、300枚/クラス) -> 既存モデル (best_resnet18_bus.pth) を流用
- 評価データ:
  - real_recaptcha (本物画像、正例6693/負例6693)
- 出力:
  - 画像分類/eval/results/ablation.md (比較表)
"""

import copy
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

# 必要なパスのインポート
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_augment import make_night_image, make_rainy_noise_image
from classification import collect_image_paths
from evaluate import load_labels, match_labels_to_paths

# パスはリポジトリのルート基準
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = Path(__file__).resolve().parent / "labels"
REAL_BUS_DIR = REPO_ROOT / "画像分類" / "real_recaptcha"
REAL_LABELS_PATH = LABELS_DIR / "real_recaptcha_labels.csv"

EXISTING_FT_MODEL_PATH = REPO_ROOT / "best_resnet18_bus.pth"
EXISTING_FT_CLASSES_PATH = REPO_ROOT / "best_resnet18_bus_classes.json"

ABLATION_DIR = REPO_ROOT / "dataset_ablation"
DEFAULT_SEED = 42

def compute_ap_fast(scores, labels):
    """APを高速に計算する"""
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

def prepare_ablation_datasets():
    """既存の dataset/ から原本画像をコピーして、ABLATION_DIR 内に条件 A と B のデータセットを構築する"""
    print("\n[Data Preparation] Preparing ablation datasets...")
    
    src_dataset_dir = REPO_ROOT / "dataset"
    if not src_dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found at {src_dataset_dir}")
        
    # クラス定義
    classes = ["bus", "other"]
    phases = ["train", "val"]
    
    # Ablation フォルダの作成
    for cond in ["clean", "old_filter"]:
        for phase in phases:
            for cls in classes:
                os.makedirs(ABLATION_DIR / cond / phase / cls, exist_ok=True)
                
    # 原本画像のコピー & 旧フィルタ適用
    random.seed(DEFAULT_SEED)
    
    for phase in phases:
        for cls in classes:
            src_dir = src_dataset_dir / phase / cls
            if not src_dir.exists():
                print(f"  Warning: {src_dir} does not exist. Skipping.")
                continue
                
            # 原本画像 (_original.jpg) のみ取得
            original_images = [f for f in os.listdir(src_dir) if f.endswith("_original.jpg")]
            print(f"  Copying {len(original_images)} original images for {phase}/{cls}...")
            
            for fname in original_images:
                base_name = fname[:-13] # strip "_original.jpg"
                src_path = src_dir / fname
                
                # --- 条件 A: clean (原本のみ) ---
                clean_dest = ABLATION_DIR / "clean" / phase / cls / fname
                shutil.copy(src_path, clean_dest)
                
                # --- 条件 B: old_filter (原本 + 夜 + 雨) ---
                # 原本コピー
                old_dest_orig = ABLATION_DIR / "old_filter" / phase / cls / fname
                shutil.copy(src_path, old_dest_orig)
                
                # 夜・雨画像を生成して保存
                with Image.open(src_path).convert("RGB") as img:
                    # 夜画像
                    night_img = make_night_image(img.copy())
                    night_img.save(ABLATION_DIR / "old_filter" / phase / cls / f"{base_name}_night.jpg")
                    # 雨画像
                    rain_img = make_rainy_noise_image(img.copy())
                    rain_img.save(ABLATION_DIR / "old_filter" / phase / cls / f"{base_name}_rain.jpg")
                    
    print("Ablation datasets preparation complete.")

def train_resnet_model(data_dir, output_model_path, device):
    """指定されたデータディレクトリの画像を用いて ResNet18 を学習する"""
    print(f"\n[Training] Training model using data from {data_dir}...")
    
    # 乱数シード固定
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    
    # データ前処理
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # データセット & データローダー
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'))
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    bus_idx = class_names.index('bus')
    
    # モデルの準備と全結合層付け替え
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = 10
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            tp, fp, fn = 0, 0, 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val':
                    for p, l in zip(preds, labels.data):
                        if p == bus_idx and l == bus_idx:
                            tp += 1
                        elif p == bus_idx and l != bus_idx:
                            fp += 1
                        elif p != bus_idx and l == bus_idx:
                            fn += 1
                            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            if phase == 'val':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 else 0.0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
        print(f"  Epoch {epoch+1}/{num_epochs} - Val Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {best_f1:.4f}")
        
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path} (Best Val F1: {best_f1:.4f})")

def evaluate_model_on_real(model_path, device, class_names=["bus", "other"], target="bus"):
    """指定されたモデルで real_recaptcha データセットを評価し AP を返す"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image_paths = collect_image_paths(REAL_BUS_DIR)
    label_dict = load_labels(REAL_LABELS_PATH)
    labeled_data = match_labels_to_paths(image_paths, label_dict)
    
    bus_idx = class_names.index(target)
    scores = []
    labels = np.array([is_pos for _, is_pos in labeled_data], dtype=bool)
    
    for count, (idx, _) in enumerate(labeled_data):
        path = image_paths[idx]
        image = Image.open(path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor)[0], dim=0).cpu()
        scores.append(probabilities[bus_idx].item())
        
    scores = np.array(scores)
    ap = compute_ap_fast(scores, labels)
    return ap

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # アブレーションデータセットの準備
    prepare_ablation_datasets()
    
    # 保存モデルパスの定義
    clean_model_path = REPO_ROOT / "best_resnet18_bus_ablation_clean.pth"
    oldfilter_model_path = REPO_ROOT / "best_resnet18_bus_ablation_oldfilter.pth"
    
    # 1. 条件 A (clean) の学習
    if not clean_model_path.exists():
        train_resnet_model(ABLATION_DIR / "clean", clean_model_path, device)
    else:
        print(f"\n[skip] Clean model already exists at {clean_model_path}")
        
    # 2. 条件 B (old_filter) の学習
    if not oldfilter_model_path.exists():
        train_resnet_model(ABLATION_DIR / "old_filter", oldfilter_model_path, device)
    else:
        print(f"\n[skip] Old filter model already exists at {oldfilter_model_path}")
        
    # 3. 3条件の AP 評価
    print("\n[Evaluation] Evaluating models on real_recaptcha...")
    
    ap_clean = evaluate_model_on_real(clean_model_path, device)
    print(f"  Condition A (clean) AP: {ap_clean:.4f}")
    
    ap_old = evaluate_model_on_real(oldfilter_model_path, device)
    print(f"  Condition B (old_filter) AP: {ap_old:.4f}")
    
    if not EXISTING_FT_MODEL_PATH.exists():
        print(f"Error: Existing fine-tuned model not found at {EXISTING_FT_MODEL_PATH}")
        sys.exit(1)
    ap_new = evaluate_model_on_real(EXISTING_FT_MODEL_PATH, device)
    print(f"  Condition C (new_filter) AP: {ap_new:.4f}")
    
    # レポート生成
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    md_path = results_dir / "ablation.md"
    
    # Zero-shot AP (基準) との差
    zs_ap = 0.875
    diff_clean = ap_clean - zs_ap
    diff_old = ap_old - zs_ap
    diff_new = ap_new - zs_ap
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 劣化フィルタのアブレーション実験評価結果\n\n")
        f.write("学習データの劣化条件を変えた3モデルにおける、本物 reCAPTCHA 画像での精度(AP)比較結果。\n\n")
        f.write("| 学習条件 | 学習データ | 枚数/クラス | real_recaptcha AP | 参考: zero-shot AP (0.875) との差 |\n")
        f.write("|---|---|---:|---:|---:|\n")
        f.write(f"| A: clean | 原本のみ | 100 | {ap_clean:.3f} | {diff_clean:+.3f} |\n")
        f.write(f"| B: 旧フィルタ | 原本＋夜＋雨 | 300 | {ap_old:.3f} | {diff_old:+.3f} |\n")
        f.write(f"| C: 新フィルタ | 原本＋degraded1＋degraded2 | 300 | {ap_new:.3f} | {diff_new:+.3f} |\n\n")
        f.write("> [!NOTE]\n")
        f.write("> 条件Aは学習枚数が100枚/クラスと少なく、B・C（300枚/クラス）との比較において「データ総枚数」の違いが含まれることに注意。\n")
        
    print(f"\nSaved Ablation Markdown report to: {md_path}")
    print("Ablation study evaluation completed successfully.")

if __name__ == "__main__":
    main()
