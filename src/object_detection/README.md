# object_detection

物体検出による reCAPTCHA 風タスクの実験コードです。

主な用途:

- YOLO detection の学習・推論
- COCO/BDD などのバウンディングボックス付きデータの変換
- 画像グリッドと検出ボックスの重なり判定
- マス選択クイズ向けの位置情報利用

代表的な入口:

```bash
uv run python src/object_detection/OD.py
uv run python src/object_detection/test_best_model.py
```
