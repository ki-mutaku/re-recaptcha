# src

実験コードは、扱うモデル・タスクの種類ごとに分けています。

## ディレクトリ

- `image_classification/`: ResNet18 などを使う画像分類系。画像あてクイズ、3x3タイル判定、分類用データ準備を含む。
- `object_detection/`: YOLO detection やバウンディングボックスを使う物体検出系。位置情報やグリッドとの重なり判定を含む。
- `segmentation/`: YOLOv8-seg などを使うセグメンテーション系。マスクラベル生成、霧/モザイク画像でのセグメンテーション学習を含む。

## 実行時の注意

多くのスクリプトは、リポジトリルートをカレントディレクトリとして実行する前提です。

例:

```bash
uv run python src/image_classification/classification.py --image-dir img --target bus --threshold 0.05
uv run python src/segmentation/train_yolov8_segmentation.py
```

今後のリファクタリングで、カレントディレクトリ依存を CLI 引数や設定に寄せていきます。
