# リファクタリング記録

このドキュメントは、実験コードと生成データが混在していた状態を整理するための作業ログです。

## 2026-07-13: 初期整理

### 目的

- Git で管理するものを、ソースコード・設定・ドキュメント中心に寄せる。
- 画像データ、生成ラベル、学習結果、重みファイルはローカル実験資産として扱い、原則 Git 管理から外す。
- ルート直下や `scripts/` 配下に散らばっていた実験スクリプトを、`src/` 配下の実験ライン別ディレクトリに集約する。

### Git 管理から外した生成物

以下のようなファイルは `.gitignore` 対象にし、既存の追跡も解除した。

- `.DS_Store`
- `img/`
- `busbus/`
- `img_bus_rain/`
- `src/test_images_fog/`
- `src/test_images_mosaic/`
- `src/my_recaptcha_dataset/data/`
- `src/yolo_dataset/`
- `*.pt`, `*.pth` などのモデル重み

`git rm --cached` で追跡だけ外しているため、ローカルファイル自体は削除していない。

### スクリプト配置

実験コードは `src/` 配下に集約し、モデル・タスクの種類ごとに分けた。

```text
src/
├─ image_classification/  # 画像分類・3x3タイル判定・分類用データ準備
├─ object_detection/     # YOLO detection・bbox変換・グリッド重なり判定
└─ segmentation/         # YOLOv8-seg・マスクラベル生成・劣化画像生成
```

以前の `scripts/` ディレクトリは廃止した。

### 今後の候補

- パス指定を定数や CLI 引数に寄せ、カレントディレクトリ依存を減らす。
- 共通処理を深いモジュールとして切り出す。
  - 画像収集
  - グリッド分割
  - データセット split
  - デバイス解決
  - 実験ログ出力

## 2026-07-14: image-classification-2 への統合

- `origin/main` の Refactoring (#28) を取り込み、`src/image_classification/`、`src/object_detection/`、`src/segmentation/` の3ライン構成へ統一した。
- 画像分類は分岐後に進んだ新版を採用し、旧版とのrename競合は新版側へ解決した。
- `img/`、`img_bus_rain/`、`busbus/` はローカル実験資産として保持しつつ、mainと同じくgit追跡を解除した。
- 物体検出用の `src/test_images_coco/` と、mainが追加したコンテナ・Slurm資料は維持した。
- `rough_bus.py` はmainの担当分けに従って `src/object_detection/` に配置した。

## 2026-07-14: セグメンテーションデータの分割を再現可能にする

### 目的

- 同じ元画像から作った fog/mosaic 画像が train/val/test をまたがないようにする。
- データセット再生成時に古いsplitのファイルが混ざらないようにする。
- 分割結果をmanifestとして保存し、同じ入力では同じsplitを再利用する。

### 変更内容

- 元画像ID単位で train 80%、val 10%、test 10%へ分割するようにした。
- `split_manifest.csv` に元画像ID、split、画像・ラベル内容の指紋を保存するようにした。
- 新しいデータセットは一時ディレクトリで検証してから配置するようにした。
- 既存の `yolo_dataset/` は削除せず、日時付きバックアップへ退避するようにした。
- `add_noise.py` の入出力パスを `src/` 基準に統一した。
- Slurmを含むCLI引数と `--prepare-only`、`--reshuffle` を追加した。

### 確認コマンド

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall -q src tests
uv run python src/segmentation/train_yolov8_segmentation.py --help
```

一時データを使うテストでは、同じ元画像IDの fog/mosaic が同じsplitへ入り、
既存manifestが再利用され、古いファイルが新データセットへ混入しないことを確認した。

実データを使う場合の入力は `src/test_images_fog/`、`src/test_images_mosaic/`、
`src/labels_fog/`、`src/labels_mosaic/`、出力は `src/yolo_dataset/` となる。
学習モデルのデフォルトは `yolov8n-seg.pt` で、今回の確認ではモデル学習と
実データセットの再生成は実行していない。

次に確認すべきことは、ラベル生成後に `--prepare-only` で実データセットを作り、
manifestのID重複がないことと、Ultralyticsが生成した `data.yaml` から
train/val/testを読み込めることを確認することである。
