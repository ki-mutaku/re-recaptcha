# zero-shot ResNet vs ファインチューニング後 比較

本物の評価データ（晴れ=img / 雨=img_bus_rain）に対して、
ImageNet事前学習のままのResNet（zero-shot）と、
bus/other 2クラスに微調整したResNet（fine-tuned）を、
**同じ PR-AUC (Average Precision) 軸**で比較した結果。

| データセット | 正例/負例 | zero-shot AP | fine-tuned AP | 差分(FT−ZS) |
|---|---:|---:|---:|---:|
| 晴れ (img) | 11/99 | 1.000 | 0.953 | -0.047 |
| 雨 (img_bus_rain) | 50/52 | 0.978 | 0.930 | -0.049 |

## 詳しい指標

AP だけでは見えない実用面の数値も並べる。
「適合率1.0での再現率」は、誤検出ゼロのまま何割のバスを拾えるか。
「再現率1.0での適合率」は、バスを全部拾おうとしたとき選択がどれだけ綺麗か。

| データセット | モデル | AP | 最大F1 (閾値) | 適合率1.0での再現率 | 再現率1.0での適合率 |
|---|---|---:|---:|---:|---:|
| 晴れ (img) | zero-shot | 1.000 | 1.000 (0.034) | 1.000 | 1.000 |
| 晴れ (img) | fine-tuned | 0.953 | 0.900 (0.995) | 0.818 | 0.647 |
| 雨 (img_bus_rain) | zero-shot | 0.978 | 0.935 (0.001) | 0.740 | 0.877 |
| 雨 (img_bus_rain) | fine-tuned | 0.930 | 0.842 (0.554) | 0.560 | 0.556 |

## PR曲線

### 晴れ (img)

![PR](img_zs_vs_ft_PR.png)

### 雨 (img_bus_rain)

![PR](img_bus_rain_zs_vs_ft_PR.png)

## 読み方

- AP（PR-AUCの値）が大きいほど高性能。1.0が理想。
- 差分が正なら fine-tuning で改善、負なら zero-shot の方が強い。
- 晴れと雨で差を比べると、劣化画像に対する強さの違いが分かる。
- zero-shot と fine-tuned は前処理が異なる（zero-shot=パディング / FT=Resize）。
  各モデルを学習時と同じ前処理で測ることで、公平な比較にしている。
