# 画像分類（bus を当てる班）— マニュアル

reCAPTCHA の「お題に合う画像を選べ」を、**画像分類**で解く側のまとめです。
お題は今のところ **bus（バス）** に絞っています。1枚のタイル画像を見て「バスか／そうでないか」だけを判定します。
「9マスのどこにバスがあるか」という位置当ては物体検出班の担当で、こちらでは扱いません。

はじめての人でも、このREADMEを上から順に実行すれば同じ結果まで辿り着けるように書いています。

---

## 0. 3行で言うと

- バスの判定は、実は **zero-shot（ImageNet で学習済みの ResNet をそのまま使う）だけでかなり解けます**。きれいな写真なら追加学習は要らないくらいです。
- ただし **本物の reCAPTCHA 画像（低解像度でノイズだらけ）では、微調整した FT モデルの方が上回ります**。本物での最大F1は 0.827 で、人間の正答率 約81%（文献値）を上回りました。
- 学習データに混ぜる「劣化加工（フィルタ）」は、当初の「夜・雨」をやめ、**本物に似せた劣化（低解像度＋JPEGノイズ）** に作り直しました。本物にどれだけ似ているかは **FID** という指標で数値で確認しています。

> **言葉が分からないときは** → 一番下の「用語集」を先に読んでください。zero-shot / FT / AP / F1 / FID をかみ砕いています。

---

## 1. クイックスタート（各ステップの解説つき）

### 準備するもの

- macOS か Linux（Mac は学習に MPS を使える。無くても CPU で動く、少し遅いだけ）
- `uv`（Python のパッケージ管理ツール）とネット接続
- Python 3.12（`.python-version` で固定済みなので `uv` が勝手に合わせる）

### 実行する場所

パスはすべてスクリプト自身の場所（`src/image_classification/` 基準）で解決するので、
**どのディレクトリから実行しても動きます**。本READMEのコマンド例はルートから実行する形で書いています。
下記の `data/`・`models/`・`eval/` はすべて `src/image_classification/` からの相対パスです。
データセットの配置一覧は `data/README.md` を参照。

### 手順1：依存パッケージを入れる

```bash
uv sync
```

`uv.lock` に固定されたバージョンで、torch / torchvision / numpy などを一括インストールします。
初回だけ時間がかかります。2回目以降は一瞬で終わります。

### 手順2：パイプラインを一括実行する

```bash
uv run python src/image_classification/main.py
```

この1コマンドが、下の **6ステップ** を順番に実行します。何をしているか1つずつ説明します。

| ステップ | やること | できるもの | なぜ必要か |
|---|---|---|---|
| **1. バス学習データDL+加工** | COCO 2017 からバス画像を集め、1枚を「原本／劣化1／劣化2」の3枚にする | `data/dataset/train/bus/` | FTモデルに「これがバス」と教える教材 |
| **2. other 学習データDL+加工** | 同じ加工で「バス以外」を集める | `data/dataset/train/other/` | 「これはバスじゃない」も教えないと判定できない |
| **3. train/val 分割** | 学習用と検証用に分ける（**元画像ID単位**で） | `data/dataset/train` `data/dataset/val` | 同じ景色が両方に入る「ズル（リーク）」を防ぐ |
| **4. ResNet18 学習** | バス/その他の2クラスに微調整（FT） | `models/best_resnet18_bus.pth` | これが判定モデルの本体 |
| **5. 本物 reCAPTCHA 画像DL** | HuggingFace から本物の reCAPTCHA 画像を取得 | `data/real_recaptcha/` | 「本番の絵」で評価するため |
| **6. フィルタ妥当性評価** | 合成した劣化が本物にどれだけ似ているかを FID で測る | `eval/results/filter_validity.csv` | 「その劣化加工は妥当なの?」に数値で答える |

> ステップ1と5は最初の1回だけ大きなダウンロードが走ります（COCO注釈zip 約240MB、本物データセット）。時間がかかりますが2回目以降はキャッシュされます。

### 途中からやり直したいとき

`main.py` は、成果物がもう有ればそのステップを飛ばします。やり直したいときは:

```bash
uv run python src/image_classification/main.py --from-step 4   # ステップ4（学習）からやり直す
uv run python src/image_classification/main.py --force          # 全ステップ強制で作り直す
uv run python src/image_classification/main.py --n 600          # 本物画像を各600枚に制限（軽く試したいとき）
```

### 手順3：本命の評価（zero-shot vs FT）を出す

パイプラインとは別に、モデルの強さを比べるレポートを出します。

```bash
uv run python src/image_classification/eval/make_real_recaptcha_labels.py   # 本物の正解ラベルCSVを作る（初回だけ）
uv run python src/image_classification/eval/compare_models.py               # 3つの評価データで zero-shot と FT を比較
```

結果は `src/image_classification/eval/results/` に、比較表（Markdown）とPR曲線の画像で出ます。

---

## 2. 全体の流れ（1枚の地図）

```
COCO（学習用の元画像）
   │  download_train_bus.py / download_train_other.py
   │  ＋ data_augment.py（本物寄りの劣化を追加）
   ▼
data/dataset/train/{bus,other}   ── split_train_val.py（ID単位で分割）──▶ train / val
   │  train_resnet.py（微調整）
   ▼
models/best_resnet18_bus.pth（FTモデル）
   │
   ├─ compare_models.py ─▶ zero-shot と AP 比較（img / img_bus_rain / real_recaptcha）
   ├─ filter_validity.py ─▶ フィルタが本物に似てるか（FID）
   └─ eval/demo_grid.py ─▶ 本物タイルで3×3を解くデモ（§8）
```

学習は **合成データ（COCO＋劣化）**、評価は **本物データ**。ここを混ぜないのが大事です（詳細は §3）。

---

## 3. ファイル早見表

直下が現役のパイプライン。`未使用/` は役目を終えたファイル置き場です。

| ファイル | 役割 |
|---|---|
| `main.py` | **一括実行の入口**。上の6ステップを順に呼ぶ（`--from-step` / `--force` / `--n` 対応）|
| `data_augment.py` | 劣化フィルタの共通部品。現行は本物寄りの `make_recaptcha_like_image` |
| `download_train_bus.py` | COCO からバス画像を取得し、原本＋劣化2種の計3枚にして `data/dataset/train/bus/` へ |
| `download_train_other.py` | 同じ加工で「バス以外」を `data/dataset/train/other/` へ |
| `split_train_val.py` | `data/dataset/train` を train/val に分割（**元画像ID単位**でリーク防止）|
| `train_resnet.py` | FT 本体。`models/best_resnet18_bus.pth` を出力（bus の F1 でベスト選択）|
| `classification.py` | zero-shot ResNet。学習なしで bus 確信度スコアを出す |
| `predict_recaptcha.py` | FTモデルで9マス（`data/test_images/tile_*.jpg`）を判定するデモ |
| `plot.py` | 学習ログをグラフにする |
| `eval/` | 評価一式（下表）|
| `data/real_recaptcha/` | 本物 reCAPTCHA 画像（git管理外。ステップ5が作る）|

### eval/ の中身

| ファイル | 役割 |
|---|---|
| `compare_models.py` | **本命の評価**。zero-shot と FT を同じ AP 軸で3データセット比較→レポート＋PR曲線 |
| `evaluate.py` | 1モデル単体の評価ロジック（AP・PR曲線・閾値スイープ）。`compare_models.py` が使う |
| `bootstrap_ci.py` | AP差の95%信頼区間をブートストラップで出す（「差は誤差では?」に答える）|
| `ablation.py` | 劣化条件を変えて学習し、フィルタの寄与を切り分ける予備実験 |
| `filter_validity.py` | **フィルタ妥当性の定量評価**。合成劣化が本物にどれだけ近いかを FID で測る |
| `demo_grid.py` | **3×3デモ**。本物タイルを9マスに並べFTモデルで解く（正解付き。§8）|
| `download_real_recaptcha.py` | 本物データセット（HuggingFace）を取得し `data/real_recaptcha/{bus,nonbus}/` へ |
| `make_real_recaptcha_labels.py` | `data/real_recaptcha/` のフォルダ構成から正解ラベルCSVを自動生成 |
| `labels/` | 正解ラベルCSV（`img` / `img_bus_rain` は手付け、`real_recaptcha` は自動生成）|
| `results/` | 評価レポート・グラフ・CSV の出力先 |

---

## 4. 学習データの作り方

学習データは COCO 2017 から自動で集めます（ステップ1〜3）。単体で回すなら:

```bash
uv run python src/image_classification/download_train_bus.py     # → data/dataset/train/bus/   に 900枚
uv run python src/image_classification/download_train_other.py   # → data/dataset/train/other/ に 900枚
uv run python src/image_classification/split_train_val.py        # → train 720/720・val 180/180 に分割
```

ベース画像300枚 × (原本 / 劣化1 / 劣化2) = 900枚/クラス。ファイル名の末尾は
`_original.jpg` / `_degraded1.jpg` / `_degraded2.jpg` です。
（当初はベース100枚。発表準備で300枚に増やしました。`MAX_BASE_IMAGES` で変更できます。）

### ここでハマった点（初学者がつまづくので残す）

**① 偏りの修正。** 最初は bus が素の昼間画像だけ、other だけに劣化版がある状態でした。これだとモデルが
「劣化した画像＝バスじゃない」と勘違いして、劣化したバスを取りこぼします。bus にも other と
**まったく同じ劣化**を入れ、両クラスを同数にそろえました。加工は `data_augment.py` に1つだけ置き、
bus と other が必ず同じ条件になるようにしています。

**② データリークの修正。** 原本と劣化版は同じ景色なので、ファイル単位でランダムに train/val を分けると
同じ景色が両方に入り、val（検証）の点数が甘く出ます（テストの答えを勉強してしまう状態）。
`split_train_val.py` は**元画像のID単位**でまとめて分けるので、同一景色の3枚は必ず片方だけに入ります。

**③ フィルタ切替時の混在バグ。** 劣化の種類を変えると古いサフィックス（旧 `_night`/`_rain`）の
ファイルが残って新しい生成分と混ざります。`data_augment.py` の `clean_stale_variants()` が
`download_train_*.py` 実行時に現行サフィックス以外を自動削除して防いでいます。

---

## 5. 劣化フィルタ（本物寄りに作り直した話）

学習データの「水増し（データ拡張）」に使う劣化加工です。**学習にだけ使い、評価には絶対に混ぜません。**

### なぜフィルタが要るのか

本物の reCAPTCHA 画像は低解像度でノイズだらけです。きれいな COCO 画像だけで学習すると、本物の
汚い画像に対応できません。そこで学習データにわざと「本物っぽい劣化」を加えて、汚い画像に強くします。

### 旧フィルタ（廃止）

- `make_night_image`（暗くする）/ `make_rainy_noise_image`（暗く＋白い線を描く）。
- でも本物の reCAPTCHA をよく見ると **雨だれの線も夜の暗さも入っていません**。劣化の種類が的外れでした。
- 関数は比較用に残していますが、学習データ生成では使いません。

### 新フィルタ `make_recaptcha_like_image`（現行）

本物 bus 画像（100×100）の劣化を観察して再現したものです:
1. **低解像度化**（一度小さくして戻す。細部を潰す）
2. **軽いガウシアンぼかし**
3. **彩度・コントラストを少し落とす**
4. **JPEG 低品質再圧縮**（ブロックノイズ。本物のノイズ再現に一番効く）

### 妥当性の検証（FID で数値化）

「その劣化、本物に似てるの?」に数値で答えます。**FID**（2つの画像群の特徴分布の距離。**小さいほど似ている**）を使い、
同じクリーンな bus 画像に各フィルタをかけ、本物 bus 6,693枚との距離を比べました。

```bash
uv run python src/image_classification/eval/filter_validity.py   # → eval/results/filter_validity.csv
```

| フィルタ | FID（↓小さいほど本物に近い）| clean基準との差 | 判定 |
|---|---:|---:|---|
| clean（劣化なし） | 206.7 | — | 基準 |
| old_night（暗く） | 180.6 | +26.1 | 少し近づく |
| **old_rain（暗く＋白線）** | **385.8** | **−179.1** | **逆効果**（本物に無い線で遠ざかる）|
| **new（低解像度＋ぼけ＋JPEG）** | **164.1** | **+42.6** | **最も近い＝妥当** |

雨フィルタは劣化なしより本物から遠ざかっていました。新フィルタが最も近く、作り直しは妥当だったと言えます。

> **細かい注意（聞かれたとき用）**
> - 同時に出る domainAUC（本物 vs 合成を見分ける分類器のAUC）は全フィルタで ≈1.0 に飽和します。これは劣化の差だけでなく**被写体の差**（COCOの街並みバス vs reCAPTCHA の切り抜きタイル）も含むため。FID の **相対比較**（同じ画像に劣化だけ変えて適用）が劣化妥当性の比較として有効で、絶対値の大きさは被写体ギャップを表します。
> - 標準の FID は InceptionV3 特徴を使いますが、ここでは ResNet18 特徴で代用しています。同じ画像に劣化だけ変えた相対比較なので、順位は特徴抽出器の選び方に対して頑健と考えられます。

---

## 6. 学習

```bash
uv run python src/image_classification/train_resnet.py
```

Mac の MPS で10エポック、だいたい1分。出力はリポジトリのルートに2つ:

- `models/best_resnet18_bus.pth` … 学習した重み
- `models/best_resnet18_bus_classes.json` … クラスの並び `["bus", "other"]`（推論側がどっちが bus か迷わないため）

ベストは Accuracy ではなく **bus の F1** で選びます（見逃しと誤爆のバランスを見たいため）。
元画像が少ないので、数エポックで train はほぼ満点になり val loss は上がる＝過学習気味です。
エポックを増やす意味は薄いです。

---

## 7. 評価と結果

### 評価データ（3セット・すべて本物の画像）

- `data/img/` … 晴れ（劣化なしの通常写真）。正例11 / 負例99（手付けラベル）
- `data/img_bus_rain/` … 雨（劣化ありの通常写真）。正例50 / 負例52（手付けラベル）
- `data/real_recaptcha/` … **本物 reCAPTCHA**。正例(bus)6,693 / 負例(nonbus)6,693（フォルダから自動ラベル）

> real_recaptcha は bus/nonbus を同数に揃えたバランス評価です。実際の reCAPTCHA ではバスの出現率はもっと低いので、AP の絶対値は実運用と一致しません（zero-shot と FT の比較の公平性には影響しません）。

```bash
uv run python src/image_classification/eval/compare_models.py   # → eval/results/zeroshot_vs_ft_比較.md ほか
```

### 結果（AP＝PR曲線の下の面積。1.0が満点）

| 評価データ | 正例/負例 | zero-shot AP | fine-tuned AP | 差(FT−ZS) |
|---|---:|---:|---:|---:|
| 晴れ (img) | 11/99 | 1.000 | 0.984 | −0.016 |
| 雨 (img_bus_rain) | 50/52 | 0.978 | 0.943 | −0.035 |
| **本物 (real_recaptcha)** | 6693/6693 | 0.875 | **0.906** | **+0.031** |

**読み取り。** 劣化のない通常写真（晴れ・雨）では一貫して zero-shot が勝つのに、**本物だけ FT が逆転して上回ります**。
劣化つきの水増し学習が「本物ドメインへの適応」に効いていることを示しています。
本物での **FT 最大F1 = 0.827（≒82.7%）** で、人間の正答率 約81%（文献値, Searles et al., USENIX Security 2023）を上回りました。

**統計的な裏付け。** 「差 +0.031 はただの誤差では?」に答えるため、評価画像を復元抽出で選び直して差を測り直す、を1万回繰り返す
**ブートストラップ法**（`eval/bootstrap_ci.py`）で95%信頼区間を出します。旧フィルタ(100ベース)モデルでの実績では、本物の
ΔAP = +0.021 [+0.015, +0.027] で0を跨がず、FT が zero-shot に勝つこと自体は有意でした。
（現行の300ベース新フィルタモデルでのCIは再計測が必要。モデル・dataset は git 管理外のため、学習したマシンで回すこと。）

### 指標の意味（初学者向け）

モデルは画像ごとに「バスっぽさ」を0〜1のスコアで出します。ある閾値以上を「選ぶ」とし、結果を
TP（正しく選んだ）/ FP（誤爆）/ FN（見逃し）/ TN（正しく見送り）に分けて、2つの割合を作ります。

- **適合率(precision)** = TP/(TP+FP)。選んだうち本当にバスだった割合。高いほど誤爆が少ない。
- **再現率(recall)** = TP/(TP+FN)。本物のバスのうち取りこぼさず選べた割合。高いほど見逃しが少ない。

この2つは綱引きの関係なので、1つの数で見たいとき:

- **F1** … 適合率と再現率の調和平均。両方そこそこ高いと大きい。
- **AP（PR-AUC）** … 閾値を端から端まで動かした PR 曲線の下の面積。閾値を決めずにモデル全体の力を1数で表せる。1.0が満点。

zero-shot と FT はスコアの作り方が違いますが、どちらも0〜1のバスっぽさなので同じ AP 軸で比較できます。

### アブレーション（フィルタの寄与の切り分け・予備実験）

「新フィルタは本物に似てる（FID）。じゃあ本当に精度も上がるの?」を確かめる予備実験です。
同じ原本100枚から劣化のかけ方だけ変えた3モデルを、同一マシン・同一シードで学習して比べました
（`eval/ablation.py` → `eval/results/ablation.md`）。

| 学習条件 | 学習データ | real_recaptcha AP | zero-shot(0.875)比 |
|---|---|---:|---:|
| 劣化なし | 原本のみ | 0.842 | −0.033 |
| 旧フィルタ | 原本＋夜＋雨 | 0.911 | +0.036 |
| 新フィルタ | 原本＋低解像度2種 | 0.868 | −0.007 |

分かったこと:
1. **劣化つき水増し自体は効きうる**。劣化なし（0.842）は zero-shot にも負け、劣化を足すと上がる。
2. **フィルタの種類による優劣は未決着**。この回は新フィルタ（0.868）が旧フィルタ（0.911）に負けたが、
   同じ新フィルタを別の回に測ると 0.906〜0.911 も出ており、単一シードでは AP が大きく振れる。
   **FID で見た目が本物に近いこと（§5）が、そのまま精度向上につながるとは今回は示せていません。**
   決着には同一ベース・複数シードで学習して平均で比べる必要があります（今後の課題）。

---

## 8. 3×3デモ（本命：本物タイルで9マスを解く）

本物 reCAPTCHA のタイル（`data/real_recaptcha/{bus,nonbus}/` の100×100画像）を3×3に並べ、
学習済みFTモデル（`models/best_resnet18_bus.pth`）で各マスを bus/other 判定するデモです。
タイルは自分で選ぶので**各マスの正解が既知**＝「9マス中何マス正解したか」まで言えます。

### 動かす順番

前提：学習済みモデル `models/best_resnet18_bus.pth` と本物タイル `data/real_recaptcha/{bus,nonbus}/` があること
（`main.py` を一度通していれば両方そろっています）。

```bash
# これだけでデモが動く（リポジトリのルートで）
uv run python src/image_classification/eval/demo_grid.py --seed 1
```

出力:
- `eval/results/demo_grid_input.png` … 出題（3×3グリッド）
- `eval/results/demo_grid_result.png` … 判定結果（緑枠＝モデルがバスと判定 / 赤枠＝間違い）
- コンソールに「正解グリッド」「モデルの選択」「確信度」「9マス中◯マス正解」

**発表で全マス正解する鉄板シード：1, 4, 6, 9, 14, 19**（`--threshold 0.5`）。

### 0から準備する場合

モデルやタイルが無い環境では、先に用意してからデモを実行します。

```bash
uv sync                                             # 1. 環境
uv run python src/image_classification/main.py                       # 2. 学習データ作成＋モデル学習＋本物タイル取得
uv run python src/image_classification/eval/demo_grid.py --seed 1    # 3. デモ実行
```

### オプション

```bash
uv run python src/image_classification/eval/demo_grid.py               # ランダムに1枚
uv run python src/image_classification/eval/demo_grid.py --seed 1      # 並びを固定（再現・発表用）
uv run python src/image_classification/eval/demo_grid.py --buses 4     # バスを4マスに（1-8）
uv run python src/image_classification/eval/demo_grid.py --threshold 0.5  # バス判定のしきい値
```

> 使うモデルは現行の **300ベース・新フィルタ FT**（`models/best_resnet18_bus.pth`）。
> モデルの1マスあたり正解率は約83%なので、**ランダムな並びでは毎回9/9になるとは限りません**
> （固定シードで「解ける例」を見せるのが安全。ライブなら「稀に1マス外す」と一言添える）。

### 旧デモ（参考）

古い枠組みとして、同ディレクトリの `split_recaptcha.py`（1枚の画像を3×3に分割→`data/test_images/tile_*.jpg`）＋
`predict_recaptcha.py`（各マス判定）もあります。ただし入力画像の正解が無く、上の `demo_grid.py` の方が
本物タイル・正解付きで発表向きです。

---

## 9. ゼロから再現する

中間ファイルが手元に何も無くても、同じ結果に辿り着けるように。

### 全手順

```bash
uv sync                                                     # ① 依存を入れる
uv run python src/image_classification/main.py                              # ② 学習データ〜本物DL〜フィルタ妥当性まで一括（6ステップ）
uv run python src/image_classification/eval/make_real_recaptcha_labels.py   # ③ 本物の正解ラベルCSV
uv run python src/image_classification/eval/compare_models.py               # ④ zero-shot vs FT の AP 比較
```

### どこまで固定できているか（再現性）

- **乱数シードは全部42**（`data_augment.py` / `split_train_val.py` / `train_resnet.py`）。同じ手順なら同じ分割・同じ学習。
- **パッケージは `uv.lock` 固定**。
- 固定しきれない部分:
  - COCO 側の画像URLが将来変われば取得物が変わる。
  - `data/img/` `data/img_bus_rain/` の評価ラベルは手付けなのでスクリプトからは作れない。
  - MPS と CPU で計算順が変わり、AP の下の桁が少しぶれることがある。
  - 劣化フィルタは乱数を含むので、FID の値も再実行で下1桁が少し動く。

### 注意：git 管理外の生成物

`models/best_resnet18_bus.pth`（モデル）と `data/dataset/`、`data/real_recaptcha/` は **git に入れていません**（作り直せるため）。
別のマシンで学習したモデルは手元に来ないので、**数値を出す前に「そのモデルはいつ・どのデータで学習したものか」を必ず確認**してください
（過去にこれで旧モデルと新モデルを取り違える事故がありました）。

---

## 10. 本物データセットのライセンス・倫理

- 出典: HuggingFace `nobodyPerfecZ/recaptchav2-29k`（実際の reCAPTCHA v2 デモページをスクレイピングした実画像、29,568枚・100×100）。
- ライセンス: MIT。ただし **画像は Google 所有**で、利用は **非営利・教育・研究目的に限定**。Google 非公式。
- 本リポジトリでは **画像実体を git 管理しません**（`data/real_recaptcha/` は `.gitignore` 済み）。**再配布しません**。
- 用途は研究目的であり、**CAPTCHA 突破そのものを目的としません**。

---

## 用語集

- **zero-shot** … 追加学習をせず、ImageNet で学習済みの ResNet をそのまま使う方式。
- **FT（ファインチューニング / fine-tuning）** … その ResNet を bus/other の2クラスに微調整する方式。
- **ResNet18** … 画像分類でよく使う定番のニューラルネットワーク。ここでは ImageNet 学習済みのものを使う。
- **フィルタ／劣化** … 学習データに本物っぽい劣化を加える加工（現行＝低解像度＋ぼけ＋JPEG）。
- **データ拡張（augmentation）** … 学習データの水増し。本物に似ているほど学習が効く。
- **リーク** … 検証・評価用のデータが学習に混ざり、点数が不当に良く出ること。
- **AP（PR-AUC）** … モデルの強さを0〜1の1数で表す。閾値に依存しない。1.0が満点。
- **F1** … 適合率と再現率の調和平均。両方高いと大きい。
- **FID** … 2つの画像群の特徴分布の距離。小さいほど似ている。フィルタが本物に近いかの判定に使う。
