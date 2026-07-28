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
uv run python src/segmentation/predict_grid_tiles.py
```

## ファインチューニング済みモデルで4×4タイルを選択する

`predict_grid_tiles.py` は、`train_yolov8_segmentation.py` が作成したtest splitを
ファインチューニング済みYOLOv8-segで推論します。対象クラスの予測マスクが
1画素でも重なるタイルを、左上から行優先・1始まりの番号で出力します。

```text
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15 16
```

既定設定では、mask mAP50-95が最も高かった `fog_mosaic_finetune-4` の
`best.pt` を使い、`src/yolo_dataset/images/test` の全画像から `bus` を探します。

```bash
uv run python src/segmentation/predict_grid_tiles.py
```

別の対象クラスや1枚の画像を指定する場合:

```bash
uv run python src/segmentation/predict_grid_tiles.py \
  --target car \
  --source src/yolo_dataset/images/test/fog_000000001625.jpg \
  --conf 0.25
```

出力先の既定値は `runs/segment/grid_tile_predictions/predict/` です。同名の結果が
ある場合は `predict2`、`predict3` のように連番を付け、既存結果を上書きしません。

- コンソール: 画像ごとの予測タイル、正解タイル、完全一致の成否
- `predictions.json`: 全画像のタイル番号、各タイルのマスク面積率、評価指標
- `images/`: 選択タイルを赤、予測マスクを緑、4×4境界を白で描いた画像

自動評価では `src/yolo_dataset/labels/test` のYOLOポリゴンを正解タイルへ変換し、
タイル単位の Accuracy、Precision、Recall、F1と、画像単位のExact Matchを計算します。
タイル内の微小なマスクを除外したい場合は、`--min-tile-mask-ratio 0.01` のように
最小面積率を指定できます。モデル、入力、正解ラベル、出力先はそれぞれ
`--model-path`、`--source`、`--labels-dir`、`--output-root` で変更できます。

既定設定（`bus`、`conf=0.25`、`iou=0.7`、`imgsz=640`）とCPUでtest split
100枚を確認した結果は、Accuracy 0.8169、Precision 0.8438、Recall 0.7697、
F1 0.8051、Exact Match 0.4800でした。次に確認すべきことは、`conf` と
`min-tile-mask-ratio` を変えたときの各指標を比較し、タイル選択に適した閾値を
決めることです。

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
