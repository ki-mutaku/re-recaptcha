# 画像分類（bus を当てる班）— マニュアル

reCAPTCHA の「お題に合う画像を選ぶ」課題を、**画像分類**で解く側のまとめ。
お題は今のところ **bus** に絞っている。マスの位置（座標）を出す物体検出は別の班の担当で、こちらは触らない。
1枚ごとに「バスか／そうでないか」を判定する。

> **結論（先に）**
> - bus は **zero-shot（ImageNet学習済みResNetをそのまま使う）で十分**。劣化のない通常写真の晴れ・雨では FT はむしろ少し下がる。
> - ただし **本物の reCAPTCHA 画像では FT が逆転して上回る**（後述）。
> - 学習データの劣化フィルタは「夜・雨（暗く＋白線）」をやめ、**本物寄りの劣化（低解像度化＋JPEG再圧縮）** に作り直した。本物との **FID** でこの方が妥当だと数字で確認済み。

---

## 1. クイックスタート

リポジトリの **ルート** で実行する（スクリプトは `dataset/` などを相対パスで見るため、`cd 画像分類` してからだと壊れる）。

```bash
uv sync                          # 依存をインストール（uv.lock 固定）
uv run python 画像分類/main.py    # 学習データ生成→分割→学習→本物DL→フィルタ妥当性評価 を一括実行
```

`main.py` は途中まで成果物があればスキップする。部分的にやり直したいとき:

```bash
uv run python 画像分類/main.py --from-step 4   # 4(学習)から再開
uv run python 画像分類/main.py --force         # 全ステップ強制再実行
uv run python 画像分類/main.py --n 600         # 本物画像を各600枚に制限（軽い）
```

評価レポート（zero-shot vs FT）は別途:

```bash
uv run python 画像分類/eval/make_real_recaptcha_labels.py   # 本物のラベルCSVを作る（初回のみ）
uv run python 画像分類/eval/compare_models.py               # 3データセットでAP比較
```

---

## 2. ファイル早見表

直下が現役パイプライン。`未使用/` は役目を終えたファイルの置き場。

| ファイル | 役割 |
|---|---|
| `main.py` | **一括実行の入口**。下の各スクリプトを順に呼ぶ（`--from-step` / `--force` 対応）|
| `data_augment.py` | 劣化フィルタの共通部品。学習データ生成は本物寄りフィルタ `make_recaptcha_like_image` を使う |
| `download_train_bus.py` | COCO からバス画像を取得し、原本＋劣化2種の計3枚にして `dataset/train/bus/` へ |
| `download_train_other.py` | 同じ加工で「バス以外」を `dataset/train/other/` へ |
| `split_train_val.py` | `dataset/train` を train/val に分割（**元画像ID単位**でリーク防止）|
| `train_resnet.py` | FT 本体。`best_resnet18_bus.pth` を吐く（F1でベスト選択）|
| `classification.py` | zero-shot ResNet。学習なしで bus 確信度スコアを出す |
| `predict_recaptcha.py` | FTモデルで 9マス（`test_images/tile_*.jpg`）を判定する本番デモ |
| `plot.py` | 学習ログをグラフにする |
| `eval/` | 評価一式（下表）|
| `real_recaptcha/` | 本物 reCAPTCHA 画像（git管理外。`download_real_recaptcha.py` が作る）|

### eval/ の中身

| ファイル | 役割 |
|---|---|
| `compare_models.py` | **本命の評価**。zero-shot と FT を同じAP軸で3データセット比較→レポート＋PR曲線 |
| `evaluate.py` | 1モデル単体の評価ロジック（AP・PR曲線・閾値スイープ）。`compare_models.py` が利用 |
| `download_real_recaptcha.py` | 本物データセット（HuggingFace）を取得し `real_recaptcha/{bus,nonbus}/` へ |
| `make_real_recaptcha_labels.py` | `real_recaptcha/` のフォルダ構成から正解ラベルCSVを自動生成 |
| `filter_validity.py` | **フィルタ妥当性の定量評価**。合成劣化が本物にどれだけ近いかを FID で測る |
| `labels/` | 正解ラベルCSV（`img` / `img_bus_rain` は手付け、`real_recaptcha` は自動生成）|
| `results/` | 評価レポート・グラフ・CSV の出力先 |

---

## 3. データの作り方（学習データ）

学習データは COCO 2017 から自動で集める。最初の1回だけ注釈zip（約240MB）を落とすので時間がかかる。

```bash
uv run python 画像分類/download_train_bus.py     # → dataset/train/bus/   に 900枚
uv run python 画像分類/download_train_other.py   # → dataset/train/other/ に 900枚
uv run python 画像分類/split_train_val.py        # → train 720/720, val 180/180 に分割
```

ベース画像300枚 × (原本 / 劣化1 / 劣化2) = 900枚/クラス。ファイル名のサフィックスは
`_original.jpg` / `_degraded1.jpg` / `_degraded2.jpg`。
（当初はベース100枚だったが、発表準備で300枚に増量。`MAX_BASE_IMAGES` で変更可能。）

### ここでハマった点（残しておく）

**偏りの修正。** 最初は bus が素の昼間画像だけで、other だけに劣化版があった。これだと
「劣化した画像＝バスじゃない」と覚えてしまい、劣化したバスを取りこぼす。bus にも other と
**まったく同じ劣化**を入れて、両クラスとも同数にそろえた。加工は `data_augment.py` に1つだけ置き、
bus と other が必ず同じ条件になるようにしている。

**リークの修正。** 原本と劣化版は同じ景色なので、ファイル単位でランダムに train/val を分けると
同じ景色が両方に入って val の点数が甘くなる。`split_train_val.py` は**元画像のID単位**でまとめて
分けるので、同一景色の3枚は必ず train か val のどちらか片方に入る。

**フィルタ切替時の混在バグ。** フィルタを変えると古いサフィックス（旧 `_night`/`_rain`）の
ファイルが残って新生成分と混ざる。`data_augment.py` の `clean_stale_variants()` が
`download_train_*.py` 実行時に現行サフィックス以外を自動削除して防いでいる。

---

## 4. 劣化フィルタ（重要：本物寄りに作り直した）

学習データの「水増し（augmentation）」に使う劣化加工。**学習にだけ使い、評価には混ぜない。**

### 旧フィルタ（廃止）
- `make_night_image`（暗く）/ `make_rainy_noise_image`（暗く＋白い線）。
- 本物の reCAPTCHA を見ると **雨だれの線も夜の暗さも入っていない**ので、劣化の種類が的外れだった。
- 関数は `filter_validity.py` の比較用に残してあるが、学習データ生成では使わない。

### 新フィルタ `make_recaptcha_like_image`（現行）
本物 bus 画像（100×100）の劣化を観察して再現:
1. **低解像度化**（縮小→元サイズへ戻す。細部を潰す）
2. **軽いガウシアンぼかし**
3. **彩度・コントラストを少し落とす**
4. **JPEG 低品質再圧縮**（ブロックノイズ。本物のノイズ再現に一番効く）

### 妥当性の検証（FID で定量化）
「合成劣化が本物の劣化分布にどれだけ近いか」を、ResNet18特徴の **FID**（小さいほど近い＝妥当）で測った。
クリーンな bus 画像に各フィルタをかけ、本物 bus 6,693枚との距離を比較:

| フィルタ | FID（↓本物に近い）| clean基準 | 判定 |
|---|---:|---:|---|
| clean（劣化なし） | 206.7 | — | 基準 |
| old_night（暗く） | 180.6 | +26.1 | 少し近づく |
| **old_rain（暗く＋白線）** | **385.8** | **−179.1** | **逆効果**（本物に無い線で遠ざかる）|
| **new（低解像度＋ぼけ＋JPEG）** | **164.1** | **+42.6** | **最も近い＝妥当** |

```bash
uv run python 画像分類/eval/filter_validity.py   # → eval/results/filter_validity.csv
```

> 注意。同時に出す domainAUC（本物 vs 合成を見分ける分類器の AUC）は全フィルタで≈1.0に飽和する。
> これは劣化の差だけでなく**被写体の差**（COCOの街並みbus vs reCAPTCHAの切り抜きタイル）も含むため。
> FID の**相対比較**（同じクリーン画像に劣化だけ変えて適用）は劣化妥当性の比較として有効で、
> 絶対値の大きさは被写体ギャップ（フィルタでは埋まらない部分）を表す。
> ※ 標準の FID は InceptionV3 特徴を使うが、ここでは ResNet18 特徴を使用している。同一のクリーン画像に劣化だけを変えて適用する相対比較なので、特徴抽出器の選択に対して順位は頑健と考えられる。

---

## 5. 学習

```bash
uv run python 画像分類/train_resnet.py
```

Mac の MPS で 10エポック、だいたい1分。出力はリポジトリのルートに2つ。

- `best_resnet18_bus.pth` … 重み
- `best_resnet18_bus_classes.json` … クラスの並び `["bus", "other"]`（推論側がどっちが bus か迷わないため）

ベストは Accuracy ではなく **bus の F1** で選ぶ（捕捉漏れと誤検出のバランスを見たいため）。
元画像が少ない（100ベース×3）ので、数エポックで train はほぼ満点になり val loss は上がる＝過学習気味。
エポックを増やす意味は薄い。

---

## 6. 評価と結果

### 評価データ（3セット・すべて本物画像）
- `img/` … 晴れ（劣化なし）。正例11 / 負例99（手付けラベル）
- `img_bus_rain/` … 雨（劣化あり）。正例50 / 負例52（手付けラベル）
- `画像分類/real_recaptcha/` … **本物 reCAPTCHA**。正例(bus)6,693 / 負例(nonbus)6,693（フォルダから自動ラベル）
※ real_recaptcha は bus/nonbus を 6,693枚ずつに揃えたバランス評価。実際の reCAPTCHA では bus 出現率はもっと低いため、AP の絶対値は実運用時と一致しない（ただし、zero-shot と FT の比較の公平性には影響しない）。

```bash
uv run python 画像分類/eval/compare_models.py   # → eval/results/zeroshot_vs_ft_比較.md ほか
```

### 結果（AP＝PR曲線の下の面積。1.0が満点）

| 評価データ | 正例/負例 | zero-shot AP | fine-tuned AP | 差(FT−ZS) |
|---|---:|---:|---:|---:|
| 晴れ (img) | 11/99 | 1.000 | 0.984 | −0.016 |
| 雨 (img_bus_rain) | 50/52 | 0.978 | 0.943 | −0.035 |
| **本物 (real_recaptcha)** | 6693/6693 | 0.875 | **0.906** | **+0.031** |

※ 統計的裏付け：ペアド・ブートストラップ法（`eval/bootstrap_ci.py`、10,000回リサンプル）で ΔAP の95%信頼区間を出せる。
**旧フィルタ（100ベース）学習モデル**で測った実績では、本物の ΔAP = +0.021 [＋0.015, +0.027] で0を跨がず、
FT が zero-shot に勝つこと自体は統計的に有意。**現行の300ベース新フィルタモデルでの CI は再計測が必要**
（モデル・dataset は git 管理外のため、300ベースで学習したマシンで `bootstrap_ci.py` を回すこと）。

**読み取り。** 劣化のない通常写真の晴れ・雨では一貫して zero-shot が勝つのに、**本物だけ FT が逆転して上回る**。
この逆転は旧フィルタ学習モデル（+0.021）でも新フィルタ学習モデル（100ベース +0.036 / 300ベース +0.031）でも起きており、
**「本物ドメインへの転移には FT（劣化つき水増し学習）が効く」ことは頑健**。
一方「新フィルタだから勝てた」とまでは言えない（旧フィルタとの AP 差は実行間ばらつきと同程度。下のアブレーション参照）。
フィルタが本物に近いこと自体は FID で独立に示されている（§4）。
※ 上表は学習ベース画像300枚（各900枚）版。本物での **FT 最大F1 = 0.827（≒82.7%）** で、
文献上の人間正答率 ~82% を上回った。学習時の val F1 は 0.9115（100ベース時の 0.857 から改善）。

### アブレーション（劣化フィルタの寄与の切り分け・途中経過）

学習データの劣化条件だけを変えて real_recaptcha の AP を比較した（`eval/ablation.py` → `eval/results/ablation.md`）。
**注意：当初「C=新フィルタ」として評価したモデルは、実際には6/25の旧フィルタ学習モデルだった**
（.pth と dataset/ は git 管理外で、新フィルタ再学習が別マシンでのみ行われていたため）。正しいラベルは以下。

| 学習条件 | ベース枚数 | real_recaptcha AP | 備考 |
|---|---:|---:|---|
| 劣化なし（clean のみ） | 100 | 0.842 | zero-shot (0.875) にも負ける |
| 旧フィルタ（夜＋雨）今回学習 | 100×3 | 0.911 | |
| 旧フィルタ（夜＋雨）6/25学習 | 100×3 | 0.896 | 当初「新フィルタ」と誤認していたモデル |
| 新フィルタ 6/30学習（別マシン） | 100×3 | 0.911 | compare_models の実績値 |
| 新フィルタ 300ベース（現行） | 300×3 | 0.906 | 上の結果表と同じ |

分かったこと：
1. **劣化つき水増し自体は明確に効く**（clean 0.842 → 劣化あり 0.896〜0.911。ただし枚数の交絡あり）。
2. **同一条件でも実行間で AP が ±0.01〜0.015 ぶれる**（旧フィルタ2回で 0.896 vs 0.911）。
   → 旧フィルタと新フィルタの差はこのばらつきに埋もれており、**フィルタ種類の優劣は AP では未決着**。
   決着させるには同一ベース・同一シード群で新旧を複数回学習して比較する必要がある（TODO）。

### 指標の意味（初学者向け）

モデルは画像ごとに「バスっぽさ」を 0〜1 のスコアで出す。ある閾値以上を「選ぶ」。
選んだ結果を TP（正しく選んだ）/ FP（誤爆）/ FN（見逃し）/ TN（正しく見送り）に分け、2つの割合を作る。

- **適合率(precision)** = TP/(TP+FP)。選んだうち本当にバスだった割合。高いほど誤爆が少ない。
- **再現率(recall)** = TP/(TP+FN)。本物のバスのうち取りこぼさず選べた割合。高いほど見逃しが少ない。

この2つは綱引きの関係。だから単一の数で見たいとき:

- **F1** … 適合率と再現率の調和平均。両方そこそこ高いと大きい。
- **PR-AUC（AP）** … 閾値を端から端まで動かしたPR曲線の下の面積。閾値を決めずにモデル全体の力を1数で表せる。1.0が満点。

zero-shot と FT はスコアの作り方が違うが、どちらも 0〜1 のバスっぽさなので同じ AP 軸で比較できる。

---

## 7. 9マス本番デモ（マスごと判定）

リポジトリ・ルートの `split_recaptcha.py` が1枚のreCAPTCHA画像を3×3＝9マスに切り（`test_images/tile_*.jpg`）、
`predict_recaptcha.py` が FTモデルで各マスを bus / other 判定して 3×3 グリッドで表示する。

```bash
uv run python split_recaptcha.py          # sample.jpg → test_images/tile_0..8.jpg
uv run python 画像分類/predict_recaptcha.py # 9マスを判定して⭕️/❌で表示
```

---

## 8. 本物データセットのライセンス・倫理

- 出典: HuggingFace `nobodyPerfecZ/recaptchav2-29k`（実際の reCAPTCHA v2 デモページをスクレイピングした実画像、29,568枚・100×100）。
- ライセンス: MIT。ただし **画像は Google 所有**で、利用は **非営利・教育・研究目的に限定**。Google 非公式。
- 本リポジトリでは **画像実体を git 管理しない**（`real_recaptcha/` は `.gitignore` 済み）。**再配布しない**。
- 用途は研究目的であり、**CAPTCHA 突破そのものを目的としない**。

---

## 9. ゼロから再現する

手元に中間ファイルが無くても同じ結果へ辿り着けるように。

### 必要なもの
- macOS か Linux（Mac なら学習に MPS。無ければ CPU でも可、少し遅いだけ）。
- Python 3.12（`.python-version` 固定）、`uv`、ネット接続。
- パッケージは `uv.lock` で固定（torch / torchvision / numpy / matplotlib / pillow / datasets / scipy / scikit-learn）。

### 全手順
```bash
uv sync
uv run python 画像分類/main.py                              # 学習データ〜本物DL〜フィルタ妥当性まで一括
uv run python 画像分類/eval/make_real_recaptcha_labels.py   # 本物ラベルCSV
uv run python 画像分類/eval/compare_models.py               # zero-shot vs FT のAP比較
```

### どこまで固定できているか
- シードは全部42（`data_augment.py` / `split_train_val.py` / `train_resnet.py`）。同じ手順なら同じ分割・同じ学習。
- パッケージ版は `uv.lock` 固定。
- 固定しきれない部分: COCO 側の画像URLが将来変われば取得物が変わる。`img/` `img_bus_rain/` の評価ラベルは手付けでスクリプトからは作れない。MPSとCPUで計算順が変わりAPの下の桁が少しぶれることがある。

---

## 用語集

- **zero-shot** … 学習せず、ImageNet学習済みResNetをそのまま使う。
- **FT（ファインチューニング）** … そのResNetを bus/other の2クラスに微調整する。
- **フィルタ／劣化** … 学習データに本物っぽい劣化を加える加工（現行＝低解像度＋JPEG）。
- **データ拡張(augmentation)** … 学習データの水増し。本物に似ているほど学習が効く。
- **AP（PR-AUC）** … モデルの強さを0〜1の1数で表す。1.0が満点。
- **FID** … 2つの画像群の特徴分布の距離。小さいほど似ている。フィルタが本物に近いかの判定に使う。
