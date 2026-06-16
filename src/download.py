import fiftyone as fo
import fiftyone.zoo as foz
import json  # JSONを整形するために追加
import os    # ファイルパスを扱うために追加

# reCAPTCHAでおなじみの欲しいクラスを指定
target_classes = ["bus"]

# COCOデータセットから指定したクラスを含む画像だけをダウンロード
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=target_classes,
    max_samples=500,
)

# 出力先のフォルダを指定
export_dir = "./my_recaptcha_dataset"

# ダウンロードしたデータをローカルのフォルダに出力
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.COCODetectionDataset,
)

# ---------------------------------------------------------
# 【追加部分】 出力された1行のJSONを読み込み、綺麗に整形して上書きする
# ---------------------------------------------------------
json_path = os.path.join(export_dir, "labels.json")

# 1. 1行に圧縮されたJSONデータを一度Pythonに読み込む
with open(json_path, 'r', encoding='utf-8') as f:
    coco_data = json.load(f)

# 2. indent=4（スペース4つ分の字下げ）と改行を入れて上書き保存する
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(coco_data, f, indent=4, ensure_ascii=False)

print("ダウンロードとJSONの整形（改行処理）が完了しました！")