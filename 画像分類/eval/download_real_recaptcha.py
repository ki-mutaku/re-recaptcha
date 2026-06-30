"""
本物の reCAPTCHA v2 画像データセットを取得し、bus / nonbus に分けて保存する。

データセット: nobodyPerfecZ/recaptchav2-29k
  - 実際の Google reCAPTCHA v2 デモページからスクレイピングした実画像（合成ではない）
  - 100x100 RGB / マルチラベル（labels は5次元 multi-hot）
  - 列インデックス: 0=bicycle, 1=bus, 2=car, 3=crosswalk, 4=hydrant
  - ライセンス: MIT（ただし画像は Google 所有。非営利・教育・研究目的に限定）

注意:
  - 画像実体は git 管理外（real_recaptcha/ は .gitignore 済み）。再配布しない。
  - 用途は研究目的。CAPTCHA 突破そのものを目的としない。

使い方:
  python 画像分類/eval/download_real_recaptcha.py              # 全bus＋同数nonbus
  python 画像分類/eval/download_real_recaptcha.py --n 600      # 各600枚に制限
  python 画像分類/eval/download_real_recaptcha.py --splits test  # testのみ
"""

import argparse
import os

from datasets import load_dataset

BUS_IDX = 1  # labels[1] が bus
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "real_recaptcha")
ALL_SPLITS = ["train", "validation", "test"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", default=ALL_SPLITS, choices=ALL_SPLITS,
        help="使う split（既定: 全部）",
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="bus / nonbus それぞれの最大保存枚数。0 なら全件（nonbusはbus枚数に揃える）",
    )
    args = parser.parse_args()

    bus_dir = os.path.join(OUT_ROOT, "bus")
    nonbus_dir = os.path.join(OUT_ROOT, "nonbus")
    os.makedirs(bus_dir, exist_ok=True)
    os.makedirs(nonbus_dir, exist_ok=True)

    bus_cap = args.n if args.n > 0 else float("inf")

    n_bus = 0
    n_nonbus = 0
    for split in args.splits:
        print(f"[load] split={split} を読み込み中（初回はDLに数分）...")
        ds = load_dataset("nobodyPerfecZ/recaptchav2-29k", split=split)
        for ex in ds:
            is_bus = ex["labels"][BUS_IDX] == 1
            if is_bus and n_bus < bus_cap:
                ex["image"].convert("RGB").save(
                    os.path.join(bus_dir, f"real_bus_{n_bus:05d}.png")
                )
                n_bus += 1
            elif (not is_bus) and n_nonbus < bus_cap:
                # nonbus は bus と同数に揃える（全件モードでは後でbus数に合わせる）
                ex["image"].convert("RGB").save(
                    os.path.join(nonbus_dir, f"real_nonbus_{n_nonbus:05d}.png")
                )
                n_nonbus += 1

    # 全件モード: nonbus を bus 枚数に合わせて間引く（クラス均衡）
    if args.n == 0 and n_nonbus > n_bus:
        extra = sorted(os.listdir(nonbus_dir))[n_bus:]
        for f in extra:
            os.remove(os.path.join(nonbus_dir, f))
        n_nonbus = n_bus

    print(f"[done] bus={n_bus} 枚 -> {bus_dir}")
    print(f"[done] nonbus={n_nonbus} 枚 -> {nonbus_dir}")


if __name__ == "__main__":
    main()
