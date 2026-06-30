# 未使用ファイル置き場

現役のパイプラインからは外れたが、履歴・参考用に残しているファイル。

## generate_noise_bus.py
`dataset/train/bus/` にノイズ付きバス画像を追加するスクリプト。
役割が `download_train_bus.py` と重複している（どちらも COCO からバスを取得して
夜・雨加工を施し `dataset/train/bus/` に保存する）。
現在は `download_train_bus.py` が `data_augment.py` の共通加工で夜・雨版を生成するため、
こちらは不要。追加データを増やしたいときの参考としてのみ残す。

## proto.py
ResNet18 の最終層を付け替えて転移学習する「雛形」。
`train_resnet.py` がこの役割を本実装として置き換えたため未使用。
`models.resnet18(pretrained=True)` という旧APIを使っている点にも注意
（現行は `weights=...` 指定）。設計の参考としてのみ残す。

## ex_train_bus.py
BDD100K（自動運転データセット）から「雨 or 夜 ＋ バス」の過酷な環境画像を抜き出す
スクリプト。外部に BDD100K 本体（数十GB）を別途用意しないと動かず、現役パイプラインからは
外れている。本物 reCAPTCHA 寄りの劣化は `data_augment.py` の合成フィルタで再現する方針に
切り替えたため未使用。過酷環境の実画像を足したくなったときの参考として残す。
