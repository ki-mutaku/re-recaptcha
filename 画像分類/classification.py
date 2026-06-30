"""
画像分類スクリプト (classification.py)

- モデル: ResNet18（ImageNetで学習済み）
- 役割: 画像ごとに「お題が写っている確信度スコア」を求めて判定する
- 前処理: 224×224のサイズにパディング・整形する
- 後処理: softmax確率を captcha用のおおまかなカテゴリごとに合計してスコア化し、
          スコアが閾値以上の画像を選択する
- 制約: ImageNetのラベル範囲にある対象のみ分類可能

"""

import argparse
import re
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
IMAGE_SIZE = 224
DEFAULT_IMAGE_DIR = "img_bus_rain"
DEFAULT_TARGET = "bus"

# 選択の判定に使う確信度スコアの閾値。スコアが閾値以上の画像を「お題が写っている」とみなす。
# 値を下げるほど再現率が上がり、上げるほど適合率が上がる。
DEFAULT_THRESHOLD = 0.05

# predict_image_labels で人が中身を確認する用に表示するラベル件数（判定には使わない）。
DEFAULT_DISPLAY_TOP_K = 3

# ResNetが返す細かいImageNetラベルを、captchaで使いたい大まかな分類にまとめる。
# 例: "school bus" や "minibus" は、どちらも "bus" として扱う。
COARSE_CATEGORY_KEYWORDS = {
    "bus": ["bus", "minibus", "trolleybus"],
    "car": [
        "car",
        "cab",
        "taxi",
        "limousine",
        "jeep",
        "minivan",
        "pickup",
        "sports car",
        "racer",
        "ambulance",
        "police van",
    ],
    "traffic_light": ["traffic light"],
    "bicycle": ["bicycle", "mountain bike", "tandem"],
    "fire_hydrant": ["fire hydrant", "hydrant"],
}

# 日本語のお題で実行しても、内部では英語の大分類名に変換して判定する。
TARGET_ALIASES = {
    "バス": "bus",
    "車": "car",
    "自動車": "car",
    "信号": "traffic_light",
    "信号機": "traffic_light",
    "自転車": "bicycle",
    "消火栓": "fire_hydrant",
}


def natural_sort_key(path):
    """
    ファイル名に含まれる数字を数値として扱うためのソートキー。

    通常の文字列ソートでは "10.jpg" が "2.jpg" より前に来てしまう。
    この関数を使うと "1.jpg", "2.jpg", ..., "10.jpg" の自然な順番になる。
    """
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part for part in parts]


def collect_image_paths(image_dir):
    """
    指定ディレクトリ以下にある画像ファイルをすべて取得する（サブディレクトリも含む）。

    real_recaptcha/{bus,nonbus}/ のようにクラスごとサブディレクトリに
    分かれているデータセットも1回の呼び出しで扱えるようにするため再帰的に探す。
    reCAPTCHAのタイル番号と出力インデックスが対応しやすいように、
    ファイル名は自然順で並べる。
    """
    image_dir_path = Path(image_dir)

    return sorted(
        (
            path
            for path in image_dir_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def load_resnet_classifier():
    """
    ImageNetで事前学習済みのResNet18と、分類ラベル一覧を読み込む。

    weights.meta["categories"] には、ResNetが判定できる1000種類の英語ラベルが入っている。
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()

    return model, weights.meta["categories"]


def build_preprocess():
    """
    PIL画像をResNetに入力できるTensorへ変換する処理を作る。

    画像のリサイズ自体は predict_image_labels() 側で ImageOps.pad を使って行う。
    captcha画像は対象物が端に寄ることがあるため、中央切り抜きではなく余白追加で224x224に揃える。
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict_class_probabilities(image_path, model, preprocess):
    """
    1枚の画像を分類し、全クラスのsoftmax確率(1次元Tensor)を返す。

    生の出力(logits)をsoftmaxで0〜1の確率に変換することで、
    「どのくらいの確信度でそのクラスと判断したか」を比較できるようにする。
    この確率を大分類ごとに合計したものが target_confidence_score のスコアになる。
    """
    image = Image.open(image_path).convert("RGB")
    padded_image = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), color=(0, 0, 0))
    input_tensor = preprocess(padded_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    return torch.softmax(output[0], dim=0)


def target_confidence_score(probabilities, categories, target_text):
    """
    全クラス確率のうち、お題の大分類に属するクラスの確率を合計してスコアにする。

    例: target が "bus" なら "school bus"・"minibus"・"trolleybus" など
    bus にまとめられる全クラスの確率を足し、「この画像がバスである確信度」とする。
    スコアが閾値以上かどうかで選択／非選択を決める。
    クラスの判定基準は find_coarse_categories による大分類一致＋ラベルの部分一致。
    """
    normalized_target = normalize_target(target_text)

    score = 0.0
    for index, label in enumerate(categories):
        coarse_categories = find_coarse_categories(label)
        if normalized_target in coarse_categories or normalized_target in label.lower():
            score += probabilities[index].item()

    return score


def predict_image_labels(
    image_path, model, categories, preprocess, top_k=DEFAULT_DISPLAY_TOP_K, verbose=True
):
    """
    1枚の画像を分類し、確率が高い順に top_k 件のラベルを返す（中身の確認用）。

    選択の判定そのものは確信度スコアの閾値で行う（solve_recaptcha_images 参照）。
    この関数はモデルが何を見ているかを人が確認するための補助的なもの。
    verbose=False にすると print を抑制できる。
    """
    if top_k < 1:
        raise ValueError("top_k は 1 以上を指定してください。")

    top_k = min(top_k, len(categories))

    probabilities = predict_class_probabilities(image_path, model, preprocess)

    _, top_indices = torch.topk(probabilities, k=top_k)
    labels = [categories[index.item()] for index in top_indices]
    coarse_categories = sorted(
        {
            coarse_category
            for label in labels
            for coarse_category in find_coarse_categories(label)
        }
    )

    if verbose:
        print(f"判定結果 [{image_path}]: {labels}")
        print(f"大まかな分類 [{image_path}]: {coarse_categories}")

    return labels


def normalize_target(target_text):
    """
    入力されたお題を、判定に使う内部カテゴリ名へ変換する。

    日本語のお題や大文字小文字の違いを吸収し、比較しやすくするために使う。
    """
    normalized = target_text.strip().lower()
    return TARGET_ALIASES.get(normalized, normalized)


def find_coarse_categories(label):
    """
    ResNetの細かいラベルが、どの大まかな分類に属するかを返す。

    例: "school bus" は ["bus"]、"sports car" は ["car"] を返す。
    どの大分類にも当てはまらない場合は空のリストを返す。
    """
    normalized_label = label.lower()
    matched_categories = []

    for coarse_category, keywords in COARSE_CATEGORY_KEYWORDS.items():
        if any(keyword in normalized_label for keyword in keywords):
            matched_categories.append(coarse_category)

    return matched_categories


def solve_recaptcha_images(
    image_paths, target_text, threshold=DEFAULT_THRESHOLD, verbose=True
):
    """
    画像一覧から、お題の確信度スコアが閾値以上の画像インデックス番号を返す。

    各画像のsoftmax確率を target_confidence_score でお題の大分類スコアにまとめ、
    そのスコアが threshold 以上なら「お題が写っている」とみなして選択する。
    返すインデックスはPythonのリストに合わせて0始まり。
    """
    model, categories = load_resnet_classifier()
    preprocess = build_preprocess()

    selected_indices = []

    for index, image_path in enumerate(image_paths):
        probabilities = predict_class_probabilities(image_path, model, preprocess)
        score = target_confidence_score(probabilities, categories, target_text)

        if verbose:
            print(f"スコア [{image_path}]: {target_text}={score:.3f}")

        if score >= threshold:
            selected_indices.append(index)

    return selected_indices


def parse_args():
    """
    コマンドライン引数を読み取る。
    """
    parser = argparse.ArgumentParser(
        description="imgディレクトリ内の画像から、指定した対象が写っているものを選択します。"
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help="判定する画像が入っているディレクトリ",
    )
    parser.add_argument(
        "--target", default=DEFAULT_TARGET, help="探したい対象の英語ラベル"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="お題と判定する確信度スコアの閾値（0〜1、低いほど多く選ぶ）",
    )

    return parser.parse_args()


def main():
    """
    スクリプトとして実行されたときの入口。
    """
    args = parse_args()
    image_paths = collect_image_paths(args.image_dir)

    if not image_paths:
        print(f"画像が見つかりませんでした: {args.image_dir}")
        return

    selected_indices = solve_recaptcha_images(
        image_paths=image_paths,
        target_text=args.target,
        threshold=args.threshold,
    )

    print("読み込んだ画像:", [str(path) for path in image_paths])
    print("最終的な選択結果:", selected_indices)


if __name__ == "__main__":
    main()
