import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def create_captcha_model(num_classes):
    """
    事前学習済みのResNet18を読み込み、CAPTCHAのタイル分類用に改造するための関数。
    元のResNet18は1000種類の画像を分類するように作られているが、
    今回は「車、バス、背景（何もない）」などの指定されたお題の数だけを出力するように
    最終層（脳の出力部分）だけを付け替える。
    """
    # 事前学習済み（賢い状態）のResNet18を呼び出す
    model = models.resnet18(pretrained=True)
    
    # 最終層（fc: Fully Connected layer）を今回のクラス数に合わせて書き換える
    # model.fc.in_features は直前の層からの入力数（ResNet18の場合は512）
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def train_model(model, dataloader, num_epochs=5):
    """
    用意したタイル画像と正解ラベルを使って、モデルに反復学習させるための機能。
    AIが「自分の予測」と「実際の正解」のズレ（誤差）を計算し、
    そのズレを小さくするように自分自身のパラメータを修正していく一連のサイクルを実行する。
    """
    # 誤差を計算するための関数（交差エントロピー誤差：分類タスクの定番）
    criterion = nn.CrossEntropyLoss()
    
    # モデルのパラメータを更新するための最適化アルゴリズム（Adam：学習が早く安定しやすい）
    # 学習率(lr)は、一度にどれくらいパラメータを修正するかの歩幅。
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # モデルを「学習モード」に切り替える（一部の層の挙動が学習用に変わるため必須）
    model.train()

    print("=== 学習を開始します ===")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # dataloaderには「9分割されたタイル画像」と「正解のクラス番号」がセットで入っている想定
        for inputs, labels in dataloader:
            
            # 1. 勾配の初期化：前の計算結果が残らないように、毎回リセットする
            optimizer.zero_grad()
            
            # 2. 順伝播（Forward）：現在のAIの脳みそで、画像が何であるか予測させる
            outputs = model(inputs)
            
            # 3. 誤差計算（Loss）：AIの予測（outputs）と、本当の正解（labels）のズレを計算する
            loss = criterion(outputs, labels)
            
            # 4. 逆伝播（Backward）：ズレを減らすためには、脳内のどの部分をどう直せばいいか計算する
            loss.backward()
            
            # 5. パラメータ更新（Step）：4の計算結果をもとに、実際に脳内の数値を書き換えて賢くする
            optimizer.step()
            
            running_loss += loss.item()
            
        # 1エポック（データセットを1周）終わるごとに、現在の誤差を表示する
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss / len(dataloader):.4f}")

    print("=== 学習が完了しました ===")
    return model

# ==========================================
# 実行イメージ（擬似コード）
# ==========================================
if __name__ == "__main__":
    # クラス数を定義（例：0=背景, 1=車, 2=バス, 3=信号機... など計10クラスと想定）
    NUM_CLASSES = 10
    
    # 1. モデルの準備
    my_model = create_captcha_model(num_classes=NUM_CLASSES)
    
    # 2. データの準備（※ここはダミーです。実際はステップ1で作るDataLoaderが入ります）
    # dummy_dataloader = [...] 
    
    # 3. 学習の実行（エラーにならないように一旦コメントアウトしています）
    # trained_model = train_model(my_model, dummy_dataloader, num_epochs=5)