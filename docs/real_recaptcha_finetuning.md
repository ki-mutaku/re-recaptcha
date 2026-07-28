# real_recaptchaの一部を学習へ使う実験

## 目的

合成画像だけでfine-tuningした既存ResNet18と、公開データセット由来の本物タイルを一部混ぜたResNet18を、
学習に使っていない固定testで比較する。本物ドメインの追加が精度低下を改善するかを、評価リークを避けて確認する。

この実験はローカル画像に対する研究・教育目的の分類評価であり、外部CAPTCHAサービスの操作や自動送信は行わない。

## データ分割

入力は `src/image_classification/data/real_recaptcha/{bus,nonbus}/` の各6,693枚。
`prepare_real_recaptcha_split.py` が次の比率で固定manifestを作る。

| split | 比率 | 用途 |
|---|---:|---|
| train | 20% | 学習へ追加する画像の選択元 |
| val | 10% | best epochの選択 |
| test | 70% | 最終比較だけに使用 |

生のSHA-256が同じ画像に加え、64-bit dHashのHamming距離が既定で4以下の画像をUnion-Findで同一グループへ束ねる。
割り当てはグループ単位なので、完全重複・近重複が複数splitへ漏れない。seedは42で、CSVが存在する場合は
`--force` を付けない限り再分割しない。

## 実行手順

```bash
# 固定manifestを作成。作成済みなら監査だけを行う
uv run python src/image_classification/eval/prepare_real_recaptcha_split.py

# 学習データ構成を検査
uv run python src/image_classification/train_real_recaptcha.py --dry-run

# 長時間処理: 合成trainに本物train各600枚を追加し、本物val各300枚でbest epochを選ぶ
uv run python src/image_classification/train_real_recaptcha.py

# held-out testだけで既存モデルと比較
uv run python src/image_classification/eval/compare_real_recaptcha_training.py
```

実装上、学習スクリプトはmanifestのtest行をDatasetへ渡さない。既存の合成のみモデル
`models/best_resnet18_bus.pth` は維持し、追加実験は次の別ファイルへ保存する。

- 重み: `models/best_resnet18_bus_mixed_real.pth`
- クラス順: `models/best_resnet18_bus_mixed_real_classes.json`
- 学習条件: `models/best_resnet18_bus_mixed_real_metadata.json`

## 比較指標と出力

比較対象は固定testのbus/nonbus同数で、両モデルにまったく同じ画像順を使う。
AP、最大F1、その閾値、accuracy@0.5を次へ保存する。

- `src/image_classification/eval/results/real_training_comparison.csv`
- `src/image_classification/eval/results/real_training_comparison.md`

比較前のsplit監査だけを行う場合は `--audit-only` を使う。短時間の動作確認では
`--max-per-class 100` のように評価枚数を制限できるが、最終値には全testを使う。

## 2026-07-14 事前監査

ローカルの13,386枚を既定条件で分割し、次を確認した。

- 画像数: bus 6,693 / nonbus 6,693
- 近重複グループ数: 13,251
- train: bus 1,339 / nonbus 1,339
- val: bus 669 / nonbus 669
- test: bus 4,685 / nonbus 4,685
- 複数splitへまたがるグループ: 0
- bus/nonbusをまたがる近重複グループ: 0
- 学習dry-run: 合成train 1,164枚 + 本物train 1,200枚、本物val 600枚

比較スクリプトは各クラス2枚のtestを使ったsmoke testまで完了した。実モデル同士の最終比較は、
長時間のmixed-real学習を実行してから全test 9,370枚で行う。

## 解釈上の注意

- 本物trainを追加したモデルがtestで上回った場合、本物ドメインを少量学習する効果を示す。
- valで設定やepochを選んだ後にtest結果を見て設定を繰り返し変更すると、testへの間接的な過適合になる。
- `real_recaptcha` はbus/nonbusを均衡化しているため、実際の出現率を反映したaccuracyではない。モデル比較はAPを主指標とする。
- 単一seedの差だけで一般化せず、次段階では複数seedの平均と信頼区間を確認する。
