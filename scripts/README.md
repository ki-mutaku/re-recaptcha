# scripts

ルート直下に散らばっていた実験用スクリプトを用途別に集約したディレクトリです。

## ディレクトリ

- `classification/`: ResNet18 などの画像分類、3x3 タイル判定系
- `dataset/`: COCO 画像取得、ノイズ付与、train/val 分割などのデータ準備系
- `analysis/`: 実験結果の確認・可視化系
- `prototypes/`: 試作や一時確認用

## 実行時の注意

多くのスクリプトは、移動前と同じくリポジトリルートをカレントディレクトリとして実行する前提です。

例:

```bash
uv run python scripts/classification/split_recaptcha.py
uv run python scripts/classification/train_resnet.py
```

今後のリファクタリングで、カレントディレクトリ依存を CLI 引数や設定に寄せていきます。
