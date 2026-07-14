# segmentation

セグメンテーションによる reCAPTCHA 風タスクの実験コードです。

主な用途:

- YOLOv8-seg の学習・推論
- マスク/ポリゴン形式のラベル生成
- 霧・モザイクなど CAPTCHA 風の画質劣化画像の生成
- マス選択クイズ向けの物体領域推定

代表的な入口:

```bash
uv run python src/segmentation/add_noise.py
uv run python src/segmentation/extract_and_apply_labels.py
uv run python src/segmentation/train_yolov8_segmentation.py
```
