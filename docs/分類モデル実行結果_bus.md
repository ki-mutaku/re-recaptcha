# 分類モデル実行結果: bus

分類モデルがどこまでできるかを確認するため、現在の `classification.py` を実行した。

## 実行条件

```bash
uv run python src/classification.py
```

- 画像ディレクトリ: `img`
- お題: `bus`
- top-k: `3`
- 判定方法: ResNet18の上位3ラベルを見て、`bus` 系の大まかな分類に入る画像を選択

## 最終的な選択結果

```text
[0, 2, 3, 4, 5, 6, 7, 26, 31, 57, 67]
```

インデックスは、`classification.py` が読み込んだ画像一覧に対する0始まりの番号。

## 選択された画像

| index | image | path | top-3 labels |
|---:|---|---|---|
| 0 | <img src="../img/1.jpg" width="160"> | `img/1.jpg` | `minibus`, `police van`, `passenger car` |
| 2 | <img src="../img/3.jpg" width="160"> | `img/3.jpg` | `trolleybus`, `minibus`, `passenger car` |
| 3 | <img src="../img/4.jpg" width="160"> | `img/4.jpg` | `minibus`, `streetcar`, `trolleybus` |
| 4 | <img src="../img/5.jpg" width="160"> | `img/5.jpg` | `garbage truck`, `minibus`, `minivan` |
| 5 | <img src="../img/6.jpg" width="160"> | `img/6.jpg` | `minibus`, `minivan`, `police van` |
| 6 | <img src="../img/7.jpg" width="160"> | `img/7.jpg` | `minibus`, `recreational vehicle`, `police van` |
| 7 | <img src="../img/8.jpg" width="160"> | `img/8.jpg` | `minibus`, `moving van`, `gas pump` |
| 26 | <img src="../img/000000001584.jpg" width="160"> | `img/000000001584.jpg` | `trolleybus`, `passenger car`, `minibus` |
| 31 | <img src="../img/000000002006.jpg" width="160"> | `img/000000002006.jpg` | `trolleybus`, `streetcar`, `minibus` |
| 57 | <img src="../img/000000005037.jpg" width="160"> | `img/000000005037.jpg` | `streetcar`, `trolleybus`, `passenger car` |
| 67 | <img src="../img/000000006040.jpg" width="160"> | `img/000000006040.jpg` | `streetcar`, `passenger car`, `trolleybus` |

## 見えたこと

分類モデルでも、画像全体がバスやバスに近い乗り物として見える場合は選択できた。

一方で、`passenger car` や `streetcar` のような近いラベルも多く出ている。
これは、分類モデルが画像内の物体を個別に検出しているわけではなく、画像全体を1つの分類として見ているためだと考えられる。

複数の物体が写っている画像では、お題の物体が写っていても上位ラベルに出ない可能性がある。
そのため、この結果は「分類モデルでどこまで選べるか」の確認として使い、次の段階では物体検出モデルと比較するのがよい。
