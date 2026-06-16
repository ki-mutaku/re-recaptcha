import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- 設定 ---
# 1. 保存した最高精度のモデル（AIの脳みそ）
MODEL_PATH = "best_resnet18_bus.pth"

# 2. テスト画像の入っているフォルダ
TEST_DIR = "test_images"

# 3. クラス名（アルファベット順で学習したため、0がbus、1がother）
CLASSES = ['bus', 'other']
# -----------

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用デバイス: {device}\n")

    # 1. AIの脳みそ（モデル）の準備と読み込み
    print("AIのモデルを読み込んでいます...")
    # 空のResNet18を用意
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASSES))
    
    # 育てた重みを注入
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval() # 学習モードから「テスト（推論）モード」へ切り替え

    # 2. 画像の前処理（学習の時のValと全く同じ条件にする必要があります）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. 9枚の画像を順番に判定
    print("\nテスト開始！reCAPTCHAの判定を行います...\n")
    results = []

    for i in range(9):
        img_path = os.path.join(TEST_DIR, f"tile_{i}.jpg")
        if not os.path.exists(img_path):
            results.append("エラー")
            continue

        # 画像を読み込んでAIが理解できるテンソル形式に変換
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # AIに判定させる（勾配計算をオフにして高速化）
        with torch.no_grad():
            outputs = model(input_tensor)
            # Softmax関数で「何%の確率でそう思っているか」を計算
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        predicted_class = CLASSES[predicted_idx.item()]
        conf_percent = confidence.item() * 100
        
        # 結果の判定
        is_bus = (predicted_class == 'bus')
        mark = "⭕️" if is_bus else "❌"
        results.append((mark, conf_percent, predicted_class))
        
        # 1枚ずつの詳細な自信度合いを表示
        print(f"マス {i+1} (tile_{i}.jpg): {predicted_class.upper():<5} (自信度: {conf_percent:5.1f}%) - {mark}")

    # 4. 視覚的に分かりやすい3x3グリッドで最終結果を表示
    print("\n" + "="*40)
    print("【AIの最終回答（reCAPTCHAシミュレーション）】")
    print(" ⭕️ = バスのマス ／ ❌ = それ以外のマス")
    print("="*40 + "\n")
    
    for row in range(3):
        line = ""
        for col in range(3):
            idx = row * 3 + col
            mark = results[idx][0]
            line += f"  [{mark}]  "
        print(line + "\n")

if __name__ == "__main__":
    main()