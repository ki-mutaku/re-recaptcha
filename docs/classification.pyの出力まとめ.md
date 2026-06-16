# classification.pyの出力まとめ

`src/classification.py` の変更によって、判定対象の画像や判定結果を確認しやすくなった。

## 実行例

```bash
python3 src/classification.py --image-dir img --target bus --top-k 5
```

## 出力できるようになったもの

### 各画像の判定結果

画像ごとに、ResNetが予測した上位ラベルを出力する。

```text
判定結果 [img/1.jpg]: ['school bus', 'minibus', 'trolleybus']
```

`--top-k` を変えることで、上位何件まで表示・判定に使うかを変更できる。

```bash
python3 src/classification.py --top-k 10
```

これにより、1位の予測だけでは外れてしまう画像でも、上位候補にお題が含まれていれば選択できる。

### 読み込んだ画像一覧

`img/` など指定したディレクトリ内の画像をすべて読み込み、実際に判定した画像一覧を出力する。

```text
読み込んだ画像: ['img/1.jpg', 'img/2.jpg', 'img/3.jpg', 'img/4.jpg']
```

以前のように `img/1.jpg` から `img/9.jpg` までをコードに直接書く必要はない。
`img/10.jpg` のようなファイルも自動で対象になる。

### 最終的な選択結果

お題に一致すると判定された画像のインデックス番号を出力する。

```text
最終的な選択結果: [0, 3, 5]
```

インデックスは0始まり。
たとえば `0` は、読み込んだ画像一覧の先頭の画像を表す。

## 実行時に変えられるパラメータ

### `--image-dir`

判定する画像が入っているディレクトリを指定する。

```bash
python3 src/classification.py --image-dir img
```

### `--target`

探したい対象の英語ラベルを指定する。

```bash
python3 src/classification.py --target bus
```

`school bus` のように、予測ラベルの一部に `bus` が含まれる場合も一致として扱う。

### `--top-k`

各画像について、予測ラベルの上位何件までを見るかを指定する。

```bash
python3 src/classification.py --top-k 5
```

デフォルトは `3`。

## 画像の並び順について

画像はファイル名の自然な順番で処理される。

```text
img/1.jpg, img/2.jpg, ..., img/9.jpg, img/10.jpg
```

通常の文字列ソートだと `img/10.jpg` が `img/2.jpg` より前に来てしまうため、
`natural_sort_key` で数字部分を数値として扱っている。
