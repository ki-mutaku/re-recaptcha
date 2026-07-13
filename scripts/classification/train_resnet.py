import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import time

# --- 設定 ---
DATA_DIR = 'dataset'
NUM_EPOCHS = 10         # 学習を繰り返す回数
BATCH_SIZE = 32         # 1回にAIに見せる画像の枚数
LEARNING_RATE = 0.0001   # 学習率（AIの学習スピード）
# -----------

def main():
    # 1. デバイスの確認（MacのGPU「MPS」が使えるかチェック）
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用するデバイス: {device}\n")

    # 2. 画像の前処理（ResNetが読み込める224x224サイズに変換・正規化）
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # 左右反転でさらにデータを水増し
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 3. データセットとデータローダーの作成
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True) 
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    print(f"認識するクラス: {class_names}")
    
    # "bus"クラスのインデックスを取得（適合率・再現率の計算に使用）
    bus_idx = class_names.index('bus')

    # 4. ResNet-18モデルの読み込みと改造
    print("ResNet-18モデルを準備中...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 最後の全結合層を2クラス（bus, other）分類用に付け替える
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 損失関数と最適化手法の定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 学習ループ
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n学習を開始します...")
    for epoch in range(NUM_EPOCHS):
        print(f'-' * 30)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        # 各エポックで学習（train）と検証（val）を交互に行う
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 評価指標用
            tp, fp, fn = 0, 0, 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 順伝播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 学習時のみ逆伝播（パラメータ更新）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計情報の計算
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 検証フェーズでTP, FP, FNをカウント
                if phase == 'val':
                    for p, l in zip(preds, labels.data):
                        if p == bus_idx and l == bus_idx:
                            tp += 1  # 真陽性
                        elif p == bus_idx and l != bus_idx:
                            fp += 1  # 偽陽性
                        elif p != bus_idx and l == bus_idx:
                            fn += 1  # 偽陰性

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase.capitalize()} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            # 検証フェーズの詳細結果とベストモデルの保存
            if phase == 'val':
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                print(f'      └─ [Metrics] TP:{tp} FP:{fp} FN:{fn}')
                print(f'      └─ 適合率(Precision): {precision:.4f}')
                print(f'      └─ 再現率(Recall):    {recall:.4f}')

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("      ★ ベストモデルを更新しました！")

    # 6. 学習完了とモデルの保存
    time_elapsed = time.time() - since
    print('-' * 30)
    print(f'学習完了! 所要時間: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最高正解率 (Val Accuracy): {best_acc:4f}')

    # 最も成績の良かった重みをロードしてファイルに保存
    model.load_state_dict(best_model_wts)
    save_path = 'best_resnet18_bus.pth'
    torch.save(model.state_dict(), save_path)
    print(f'ベストモデルを "{save_path}" に保存しました。')

if __name__ == '__main__':
    main()