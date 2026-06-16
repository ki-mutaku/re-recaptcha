# 画像分類タスク

import torch
import glob
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


def load_resnet_model():
    """
    画像分類を行うための事前学習済みResNetモデルと、クラス名（ラベル）を読み込む機能。
    """
    # 警告文（UserWarning）を解消するための新しい書き方
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    # ImageNetの1000種類のカテゴリ名（英語）のリストを取得する
    categories = weights.meta["categories"]

    return model, categories


def predict_single_image(image_path, model, categories):
    """
    1枚の画像データを受け取り、何が写っているかをResNetを用いて予測し、文字列（英語）で返す機能。
    """
    image = Image.open(image_path).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    # 1000個のクラスの中で最も確率が高いもののインデックス（番号）を取得
    _, top_class_idx = torch.max(output, 1)

    # インデックスから具体的なカテゴリ名（テキスト）を取得
    predicted_text = categories[top_class_idx.item()]

    # デバッグ用：AIが各画像を何だと判定したかターミナルに出力して確認できるようにする
    print(f"判定結果 [{image_path}]: {predicted_text}")

    return predicted_text


def solve_recaptcha_9_images(image_paths, target_text):
    """
    9枚の画像パスの配列とお題テキストを受け取り、お題を含む画像のインデックス番号を配列で出力する機能。
    """
    model, categories = load_resnet_model()
    result_indices = []

    for index, image_path in enumerate(image_paths):
        prediction = predict_single_image(image_path, model, categories)

        # お題の文字列（例："bus"）が予測結果に含まれているかチェック
        # 大文字小文字の違いを無視するために .lower() を使う
        if target_text.lower() in prediction.lower():
            result_indices.append(index)

    return result_indices


# --- 実行部分 ---
if __name__ == "__main__":
    # お手元の画像パスのリストに書き換えてください
    image_paths = sorted(glob.glob("busbus/*.jpg"))

    # "bus" を探す
    result = solve_recaptcha_9_images(image_paths, "bus")

    print("最終的な選択結果:", result)
