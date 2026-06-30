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
  python 画像分類/eval/download_real_recaptcha.py            # test split から各 600 枚
  python 画像分類/eval/download_real_recaptcha.py --n 1000 --split test
"""

import argparse
import os

from datasets import load_dataset

BUS_IDX = 1  # labels[1] が bus
OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "real_recaptcha")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=600, help="bus / nonbus それぞれの最大保存枚数")
    args = parser.parse_args()

    bus_dir = os.path.join(OUT_ROOT, "bus")
    nonbus_dir = os.path.join(OUT_ROOT, "nonbus")
    os.makedirs(bus_dir, exist_ok=True)
    os.makedirs(nonbus_dir, exist_ok=True)

    print(f"[load] nobodyPerfecZ/recaptchav2-29k split={args.split} を読み込み中（初回はDLに数分）...")
    ds = load_dataset("nobodyPerfecZ/recaptchav2-29k", split=args.split)

    n_bus = 0
    n_nonbus = 0
    for i, ex in enumerate(ds):
        is_bus = ex["labels"][BUS_IDX] == 1
        img = ex["image"].convert("RGB")
        if is_bus and n_bus < args.n:
            img.save(os.path.join(bus_dir, f"real_bus_{n_bus:04d}.png"))
            n_bus += 1
        elif (not is_bus) and n_nonbus < args.n:
            img.save(os.path.join(nonbus_dir, f"real_nonbus_{n_nonbus:04d}.png"))
            n_nonbus += 1
        if n_bus >= args.n and n_nonbus >= args.n:
            break

    print(f"[done] bus={n_bus} 枚 -> {bus_dir}")
    print(f"[done] nonbus={n_nonbus} 枚 -> {nonbus_dir}")


if __name__ == "__main__":
    main()
