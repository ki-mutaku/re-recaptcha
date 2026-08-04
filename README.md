# re-recaptcha

reCAPTCHA風の画像選択タスクを題材に、画像分類・物体検出・セグメンテーションを
比較する研究・教育目的の機械学習実験リポジトリです。ローカル画像、公開データセット、
自作の合成データだけを扱い、実サービスのCAPTCHA操作や認証回避は行いません。

## 実験ライン

- [画像分類](src/image_classification/): ResNet18を使ったbus / otherの二値分類、データ生成、評価
- [物体検出](src/object_detection/): YOLO detectionとバウンディングボックスを使った位置推定
- [セグメンテーション](src/segmentation/): YOLOv8-segと予測マスクを使った4×4タイル選択

コード全体の配置は [`src/README.md`](src/README.md)、Singularity / Slurmでの実行方法は
[`docs/singularity_slurm.md`](docs/singularity_slurm.md) を参照してください。

## 定義

このリポジトリでは、reCAPTCHA風タスクを次の2形式に分けて扱います。

1. 9枚の異なる画像から、お題に合致する画像をすべて選択する（画像選択）
2. 1枚の画像を3×3や4×4などに分割し、お題の物体を含むマスをすべて選択する（マス選択）

画像選択は主に画像分類、マス選択は物体検出とセグメンテーションで実験します。

## 実験の進め方

### 基本的な流れ

1. タスクをGithub Issueに登録
2. 概要や完了条件など必要な情報を記入する
3. タスクを始める時はProjectsにて作業中カテゴリに移動させる
4. 作業中にやったことや気づいたこと、疑問点等はIssueのコメント欄に記入
5. タスクが終了したら完了カテゴリに移動

### 時間があればやってほしいこと

- ProjectsのPriorityとSizeを設定する
- ProjectsのRoadmapでタスクの開始日と終了日を設定する
