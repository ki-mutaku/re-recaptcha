# 画像分類（bus / other）

reCAPTCHA風の「9枚の画像から、お題（バス）に合うものをすべて選ぶ」クイズを、1枚ずつの二値分類（bus / other）として解く実験ラインです。モデルはImageNet事前学習済みのResNet18を使います。

扱うのは、ローカル画像・公開データセット・自作の合成データに対する研究・教育目的の実験だけです。実サービスのCAPTCHA操作、自動送信、認証回避はやりません。「1枚の大きな画像をマスに割って、対象物を含むマスを選ぶ」形式は `src/object_detection/` と `src/segmentation/` の担当です。

## 読み方

- 手元で0から動かしたい人 → [0からの再現手順](#0からの再現手順) を上から順に実行してください。
- 実験の現状と数字を知りたい人 → [実験方式と現在地](#実験方式と現在地) と [記録済みの結果](#記録済みの結果) へ。

## 最低限の用語

| 用語 | 意味 |
|---|---|
| zero-shot | 追加学習なしで、ImageNet学習済みResNet18をそのまま使う方式 |
| FT（fine-tuning） | 学習済みモデルを、自分のデータで少しだけ再学習する方式 |
| 合成データ | COCOの写真にreCAPTCHA風の劣化（低解像度化、ぼかし、彩度低下、JPEG再圧縮）をかけて作った学習データ |
| 本物データ | 実際のreCAPTCHA v2から集められた公開データセット（HuggingFace `nobodyPerfecZ/recaptchav2-29k`、100×100タイル約29,000枚） |
| train / val / test | 学習用 / 学習途中の答え合わせ用 / 最後に1回だけ測る用。testの画像が学習側に混ざると点が甘く出ます（リーク） |
| F1・AP | 分類の良さを0〜1で表す指標。1が満点。詳しくは `eval/results/` の各レポート参照 |

## 0からの再現手順

前提はPython 3.12と`uv`です。コマンドはすべてリポジトリのルートで実行します。COCOアノテーション（約250MB）、HuggingFaceデータセット、ResNet18の重みを取得するため、ネット接続が必要です。

### 手順0: 依存関係を入れる

```bash
uv sync --frozen
```

以降のコマンドにも `--frozen` を付けます。実行のたびに `uv.lock` が意図せず書き換わるのを防ぐためです。

### 手順1: 合成学習データを作る

```bash
uv run --frozen python src/image_classification/download_train_bus.py
uv run --frozen python src/image_classification/download_train_other.py
uv run --frozen python src/image_classification/split_train_val.py
```

- 1本目は、COCOのアノテーションを自動ダウンロードし、バスが画面の10%以上を占める画像を最大300枚選びます。1枚ごとに `_original.jpg`（元画像）、`_degraded1.jpg`、`_degraded2.jpg`（同じ劣化処理を別の乱数で2回）の3枚を保存するので、`data/dataset/train/bus/` に最大900枚できます。
- 2本目は、バス以外を同じ形式で最大300枚集めます。半分はtruckやcarといったバスと紛らわしい画像、残りはpersonやdogなど明らかに違う画像です。保存先は `data/dataset/train/other/`。
- 3本目は、各クラスの約20%を `data/dataset/val/` へ移します。同じ元画像から作った3枚は必ずtrainかvalの片方へまとめ、同じ景色が両方に入るリークを防ぎます。

### 手順2: 学習する（synthetic-only FT）

```bash
uv run --frozen python src/image_classification/train_resnet.py
```

ResNet18の全層を10 epoch微調整し、valのbus F1が最大だったepochの重みを保存します。シードは42固定。MacならMPS（GPU）を自動で使い、無ければCPUで動きます。

出力:

- `models/best_resnet18_bus.pth`
- `models/best_resnet18_bus_classes.json`

### 手順3: 本物reCAPTCHA画像を取る

```bash
uv run --frozen python src/image_classification/eval/download_real_recaptcha.py
```

`data/real_recaptcha/{bus,nonbus}/` に100×100のタイルが各6,693枚保存されます。画像はGoogle所有のため、コミットも再配布もしません（`.gitignore` 済み）。用途は非営利の研究・教育に限ります。

### 手順4: 3×3デモを動かす

```bash
uv run --frozen python src/image_classification/eval/demo_grid.py --seed 1
```

本物タイルを3×3に並べ、各マスを手順2のモデルに判定させます。`--buses 4` でバスのマス数、`--threshold 0.5` で判定しきい値、`--seed` で並びを固定できます。出力は毎回同じ2ファイルへ上書きされます。

- `eval/results/demo_grid_input.png`（出題グリッド）
- `eval/results/demo_grid_result.png`（判定結果。緑枠がバス判定、赤枠が間違い）

このデモは `real_recaptcha` 全体からタイルを選ぶので、正式な性能評価には使いません。数値は後述の固定testで測ります。

### 一括実行したい場合（main.py）

`main.py` は手順1〜3と学習、フィルタ妥当性評価を順に呼びます。

```bash
uv run --frozen python src/image_classification/main.py
```

ただし注意が2つあります。

- 最後のstep 6（フィルタ妥当性評価）はGit管理外の `data/busbus/` が必要なので、fresh cloneではstep 5までしか完走しません。
- 各工程は対象ファイルが1枚でもあると完了扱いでスキップします。途中で止まった生成物を検出できません。`--force` も既存のtrain / valを空にしてから作り直す処理ではないため、大事なローカル資産があるマシンでは使わないでください。

### ダウンロードでは作れないデータ

以下は既存のローカル資産です。無くても手順1〜4は完走できます。配置の全体像は [`data/README.md`](data/README.md) が正です。

| パス | 内容 | 使い道 |
|---|---|---|
| `data/img/` | 晴れの通常写真110枚 | 旧評価セット |
| `data/img_bus_rain/` | 雨の写真102枚 | 旧評価セット |
| `data/busbus/` | クリーンなbus原本95枚 | フィルタ妥当性評価（FID）の適用元 |

## 実験方式と現在地

2026-07-14時点で、次の3方式を区別しています。

| 方式 | 学習データ | 評価 | 状態 |
|---|---|---|---|
| zero-shot | ImageNetの事前学習のみ | 手動ラベル付き画像 | 実装・過去評価あり |
| synthetic-only FT | COCO画像と合成劣化 | 手動ラベル付き画像、`real_recaptcha` 全体 | 実装・過去評価あり |
| mixed-real FT | 合成画像＋固定splitの本物train | 近重複を除いた固定test | split・学習コード・監査まで完了。本学習と最終比較は未実行 |

ローカルにある現在のsynthetic-onlyモデルは、2026-07-14にbus 185原本、other 300原本から再学習したものです（0から再現した場合の枚数とは異なります）。最高val bus F1は `0.8761` ですが、このモデル自体のheld-out real test性能は未計測です。過去レポートの本物データに対する `AP=0.906`、最大`F1=0.827` は別の学習時点の記録で、現在の重みの性能としては扱いません。

また、タイル1枚単位のF1と、人間が9マスすべてを正しく選ぶクイズ1問単位の正答率は別の指標です。数値だけを並べて「人間を上回った」とは結論しません。

## 本物画像を学習に混ぜる実験（mixed-real）

合成データだけの学習では、本物とのドメイン差が残ります（後述のdomain AUCがほぼ1.0）。そこで本物29,000枚の一部をtrainへ混ぜて精度が上がるかを試すのがこの実験です。要点は、最終評価に使うtestを先に隔離し、学習側へ一切見せないことです。設計の経緯は [docs/real_recaptcha_finetuning.md](../../docs/real_recaptcha_finetuning.md) にあります。

### 固定splitを監査する

次のコマンドは既存manifestを上書きせず、行数、splitの分離、対応画像の存在を確認します。学習は始まりません。

```bash
uv run --frozen python src/image_classification/eval/prepare_real_recaptcha_split.py
uv run --frozen python src/image_classification/train_real_recaptcha.py --dry-run
uv run --frozen python src/image_classification/eval/compare_real_recaptcha_training.py --audit-only
```

現在の固定splitは次のとおりです。正本は `eval/splits/real_recaptcha_split.csv` です。

| split | bus | nonbus | 用途 |
|---|---:|---:|---|
| train | 1,339 | 1,339 | 本物画像を学習へ追加する候補 |
| val | 669 | 669 | best epochの選択 |
| test | 4,685 | 4,685 | 最終比較専用 |

完全一致はSHA-256、見た目がほぼ同じ近重複は64-bit dHashのHamming距離4以下でグループ化し、同一グループを複数splitへ入れません。

注意: 通常の監査はパスの存在とmanifest内部の構造を確認する処理で、画像を再ハッシュして内容の変化まで検出するものではありません。

### 学習と最終比較

```bash
uv run --frozen python src/image_classification/train_real_recaptcha.py
uv run --frozen python src/image_classification/eval/compare_real_recaptcha_training.py
```

既定条件:

- synthetic train: 現在1,164枚
- real train: bus 600 + nonbus 600
- real val: bus 300 + nonbus 300
- epoch: 10、batch size: 32、learning rate: `1e-4`
- device: `CUDA → MPS → CPU` の順で自動選択

出力:

- `models/best_resnet18_bus_mixed_real.pth`（既存の合成のみモデルは上書きしません）
- `models/best_resnet18_bus_mixed_real_classes.json`
- `models/best_resnet18_bus_mixed_real_metadata.json`
- `eval/results/real_training_comparison.csv`
- `eval/results/real_training_comparison.md`

2026-07-14時点では、本学習と全test 9,370枚での比較は未実行です。

## 記録済みの結果

### 過去のzero-shot / synthetic-only FT比較

過去に `real_recaptcha` 全体で測った記録です。現在のローカル重みの再評価値ではありません。

| データセット | zero-shot AP | synthetic-only FT AP | 差 |
|---|---:|---:|---:|
| 晴れ `img/` | 1.000 | 0.984 | -0.016 |
| 雨 `img_bus_rain/` | 0.978 | 0.943 | -0.035 |
| 本物 `real_recaptcha` | 0.875 | 0.906 | +0.031 |

詳細は [`eval/results/zeroshot_vs_ft_比較.md`](eval/results/zeroshot_vs_ft_比較.md) にあります。現在の固定testを導入する前の全体評価なので、mixed-real実験の最終値とは比較しません。

再実行するコードは次です（mixed-realモデルの評価には使いません）。

```bash
uv run --frozen python src/image_classification/eval/compare_models.py
```

### 劣化フィルタの相対比較

ResNet18の512次元特徴を使ったFrechet距離です。標準的なInceptionV3 FIDではないため、同じ実装内の相対順位として読みます。

| 条件 | real busへの距離（小さいほど近い） | domain AUC（0.5に近いほど区別困難） |
|---|---:|---:|
| clean | 206.75 | 0.9999 |
| old night | 179.91 | 1.0000 |
| old rain | 377.69 | 1.0000 |
| new | **165.22** | **0.9996** |

newフィルタは4条件の中で本物に最も近い一方、domain AUCはほぼ1.0のままです。本物と合成は依然として容易に区別できるので、「本物と同じ分布になった」ことを示す結果ではありません。

## 主なファイル

| ファイル | 役割 |
|---|---|
| `classification.py` | ImageNet ResNet18によるzero-shotスコア |
| `data_augment.py` | 現行と旧の劣化フィルタ |
| `download_train_bus.py` | COCOからbus原本を取得して劣化版を生成 |
| `download_train_other.py` | busを含まないnegativeを取得して劣化版を生成 |
| `split_train_val.py` | 元画像ID単位のtrain/val分割 |
| `train_resnet.py` | synthetic-only FT |
| `main.py` | 既存環境向け6工程オーケストレータ |
| `split_recaptcha.py` / `predict_recaptcha.py` | 1枚のスクショを9分割して判定する旧デモ |
| `eval/evaluate.py` | zero-shotの閾値スイープとAP |
| `eval/compare_models.py` | 過去方式のzero-shot / FT比較 |
| `eval/filter_validity.py` | 劣化フィルタの特徴距離とdomain AUC |
| `eval/demo_grid.py` | 3×3の定性的デモ |
| `eval/download_real_recaptcha.py` | 本物タイルの取得 |
| `eval/prepare_real_recaptcha_split.py` | 近重複を考慮した固定splitの作成と監査 |
| `train_real_recaptcha.py` | mixed-real FT |
| `eval/compare_real_recaptcha_training.py` | 固定testでの最終比較 |
| `未使用/` | 現行フローから外した旧実験コード |

## 既知の制約と次の課題

- zero-shotのbusスコアは、ImageNetラベルへの単純な部分文字列一致を含みます。`minibus`、`school bus`、`trolleybus` だけでなく `bustard`（ノガン）と `colobus`（サル）も合算されるため、ラベルID指定へ直して再評価が必要です。
- synthetic-onlyモデルには、学習条件やデータの指紋を残すmetadataがありません。ファイル名だけで性能を対応づけず、ハッシュと作業ログを確認してください。
- `main.py` のskip条件は完全性チェックではありません。期待原本数と3バリアントの整合性を検証する処理が必要です。
- 学習はseedを固定していますが、すべてのデバイスで完全な決定性を保証する設定ではありません。複数seedの平均と信頼区間が必要です。
- 旧アブレーションは生成ディレクトリを清掃せず、結果表の枚数も固定値です。再実行値を正式結果にする前に、入力件数とモデルmetadataを記録する改修が必要です。
- mixed-realモデルの本学習、固定testでの最終比較、複数seed評価は未完了です。

## 関連資料

- [本物画像を一部学習へ加える実験設計](../../docs/real_recaptcha_finetuning.md)
- [データ配置の一覧](data/README.md)
- [固定splitの説明](eval/splits/README.md)
- [2026-07-07 進捗報告](進捗報告_2026-07-07.md)
- [実験メモ](memo.md)
