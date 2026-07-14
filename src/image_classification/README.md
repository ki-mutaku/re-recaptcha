# image_classification

画像分類による reCAPTCHA 風タスクの実験コードです。

主な用途:

- ResNet18 による画像分類
- 3x3 の画像あてクイズの判定
- `bus` / `other` などの分類用データセット作成
- 分類モデルの学習結果確認

代表的な入口:

```bash
uv run python src/image_classification/classification.py --image-dir img --target bus --threshold 0.05
uv run python src/image_classification/train_resnet.py
uv run python src/image_classification/predict_recaptcha.py
```
