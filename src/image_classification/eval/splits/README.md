# 固定データ分割

`real_recaptcha_split.csv` は、ローカルの `data/real_recaptcha/` を
train / val / test に割り当てる正本です。画像本体はgit管理しませんが、分割結果と再現条件は共有します。

## 分割方針

- train: 20%。本物画像を学習へ混ぜる候補。
- val: 10%。epoch選択とモデル調整専用。
- test: 70%。最終比較専用。学習・モデル選択では読まない。
- SHA-256が同じ画像と、64-bit dHashのHamming距離が4以下の画像は同一グループにする。
- グループ単位で割り当て、近重複がsplitをまたがないようにする。

## 作成と監査

```bash
uv run python src/image_classification/eval/prepare_real_recaptcha_split.py
```

初回はCSVと `.meta.json` を作成します。2回目以降は既存CSVを変更せず、構造と画像の存在だけを監査します。
データセットを意図的に入れ替えて再分割するときだけ `--force` を指定してください。

CSVの `image_path` は `data/real_recaptcha/` からの相対パスです。
`.meta.json` にはseed、分割比、画像・グループ数、データセットfingerprint、manifest SHA-256を記録します。
