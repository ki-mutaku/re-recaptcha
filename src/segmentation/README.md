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

## データセット生成の流れ

各スクリプトはカレントディレクトリではなく `src/` を基準に入出力先を解決します。

1. `add_noise.py` が元画像から `test_images_fog/` と `test_images_mosaic/` を生成する。
2. `extract_and_apply_labels.py` が `labels_fog/` と `labels_mosaic/` を生成する。
3. `train_yolov8_segmentation.py` が元画像ID単位で train/val/test に分割する。

同じ元画像の fog/mosaic 版は必ず同じ split に入り、割り当ては
`src/yolo_dataset/split_manifest.csv` に保存されます。manifestには各IDの
画像・ラベル内容から計算した指紋も記録されます。デフォルトの比率は train 80%、
val 10%、test 10%です。

同じ入力データで再実行した場合は既存manifestを読み込み、過去と同じsplitを
再利用します。入力画像の増減や内容変更があった場合は安全のため処理を中断します。
変更後のデータを意図的に再分割するときだけ `--reshuffle` を指定してください。

データセットだけ準備する場合:

```bash
uv run python src/segmentation/train_yolov8_segmentation.py --prepare-only
```

既存の `src/yolo_dataset/` がある場合は削除せず、日時付きの
`src/yolo_dataset.backup-*` に退避してから新しいデータセットへ入れ替えます。
内容を確認した後、不要なバックアップは手動で整理してください。

分割比率や乱数シードを変更する場合:

```bash
uv run python src/segmentation/train_yolov8_segmentation.py \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --seed 42 \
  --reshuffle \
  --prepare-only
```
