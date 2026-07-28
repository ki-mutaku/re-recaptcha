# data/ — データセット置き場

画像分類（bus判定）で使うデータはすべてこのディレクトリにまとめる。
「どれが手元にあるべきで、無ければどのスクリプトで作るか」はこの表が正。

## 一覧

| ディレクトリ | 内容 | 枚数 | git管理 | 用途 | 作り方（無い場合） |
|---|---|---|---|---|---|
| `img/` | COCO由来の劣化なし通常写真（晴れ） | 110 | ×（ローカル資産） | 評価（晴れ） | 既存ローカル資産を利用 |
| `img_bus_rain/` | 雨天の本物写真 | 102 | ×（ローカル資産） | 評価（雨） | 既存ローカル資産を利用 |
| `busbus/` | クリーンなbus原本 | 95 | ×（ローカル資産） | フィルタ妥当性評価（FID）の適用元 | 既存ローカル資産を利用 |
| `samples/sample.jpg` | reCAPTCHAスクショのサンプル画像 | 1 | ○ | `split_recaptcha.py` の入力 | コミット済み。作業不要 |
| `samples/legacy/sample_reCAPTCHA.jpg` | 旧サンプル画像 | 1 | ○ | 現行コードでは未使用（由来確認用に保管） | 新規処理には使わない |
| `real_recaptcha/{bus,nonbus}/` | 本物reCAPTCHA v2タイル（100×100） | 各6,693 | ×（再配布しない） | 本物での評価・デモ | `uv run python src/image_classification/eval/download_real_recaptcha.py` |
| `dataset/{train,val}/{bus,other}/` | 合成学習データ（COCO＋劣化フィルタ） | 約1,455 | ×（生成物） | ResNet18のFT学習 | `uv run python src/image_classification/main.py`（step 1〜3） |
| `dataset_ablation/` | アブレーション用3条件（clean/old/new） | 約1,400 | ×（生成物） | `eval/ablation.py` | `uv run python src/image_classification/eval/ablation.py` が自動生成 |
| `coco/` | COCOアノテーション（zip＋解凍後JSON） | — | ×（DL物） | 学習データ生成の元情報 | `download_train_*.py` が自動DL |
| `test_images/` | 3×3分割したタイル画像 | 9 | ×（生成物） | `predict_recaptcha.py` のデモ入力 | `uv run python src/image_classification/split_recaptcha.py` |

学習済みモデル（`best_resnet18_bus.pth` など）は画像分類ディレクトリ内の `models/` に置く（git管理外）。

## まとめて用意する

0から全部そろえるには、リポジトリのルートで:

```bash
uv run python src/image_classification/main.py
```

（DL＋加工 → train/val分割 → 学習 → 本物タイルDL → フィルタ妥当性評価まで一括。詳細は `../README.md`）

## 注意

- `real_recaptcha/` は HuggingFace `nobodyPerfecZ/recaptchav2-29k` 由来。画像はGoogle所有のためコミット・再配布しない。
- 評価用ラベルCSV（`../eval/labels/*.csv`）のキーは `img/1.jpg` のような「ディレクトリ名/ファイル名」の2階層。**この中のディレクトリ名を変えるとラベルと照合できなくなる**ので、名前は変えないこと。
- `real_recaptcha/` を学習にも使う場合は、`../eval/splits/real_recaptcha_split.csv` を固定の正本とする。画像を直接コピーして分割せず、manifestの `train` / `val` / `test` を参照する。
- git管理外のデータは別マシンには同期されない。モデルの取り違え事故（memo 2026-07-07参照）を防ぐため、数値を出す前に「そのデータ・モデルはいつどの設定で作ったか」を確認する。

## git管理と利用状況

- mainのリファクタリング方針に合わせ、画像実体は原則git管理しない。`img/`・`img_bus_rain/`・`busbus/` もローカルには残すが追跡対象外。
- 学習データ `dataset/`、アブレーション生成物 `dataset_ablation/`、COCO取得物 `coco/`、本物タイル `real_recaptcha/`、分割結果 `test_images/` も、再生成・再取得できる大容量データなのでgit管理しない。
- gitで共有するのは、コード、データ取得・分割手順、ラベルCSV、固定split manifest、実験結果の要約。デモの小さな入力サンプルだけは例外として保持する。
- 現行の `dataset/` は `_original`・`_degraded1`・`_degraded2` の3種だけで、旧 `_night`・`_rain` は混在していない。
- `dataset_ablation/` の旧フィルタ画像は比較実験用に意図して残している生成物で、通常学習には使わない。
- `samples/legacy/sample_reCAPTCHA.jpg` だけは現行コードから参照されない旧資産。削除判断ができるよう隔離し、通常のデモ入力と分けた。
