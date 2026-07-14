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
