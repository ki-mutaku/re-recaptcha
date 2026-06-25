"""
データ加工（夜・雨ノイズ）の共通モジュール。

download_train_bus.py と download_train_other.py の両方から読み込んで使う。
bus クラスと other クラスに「全く同じ」加工を適用することで、
「暗い画像／雨の画像 ＝ other」という偏った学習を防ぐのが目的。
（bus を素の昼間画像だけにすると、雨のバスを一度も学習できず誤判定する）
"""

import os
import random

from PIL import Image, ImageEnhance, ImageDraw

# 加工結果を再現可能にするための乱数シード。
# bus と other で同じ値を使うことで、毎回同じ条件のデータセットを作れる。
DEFAULT_SEED = 42


def make_night_image(img):
    """
    画像を暗く・低コントラストにして「夜間」っぽい見た目を作る。

    明るさを0.2〜0.4倍、コントラストを0.7〜0.9倍に落とす。
    値はランダムなので、毎回少しずつ違う夜画像になる。
    """
    enhancer_brightness = ImageEnhance.Brightness(img)
    img_night = enhancer_brightness.enhance(random.uniform(0.2, 0.4))
    enhancer_contrast = ImageEnhance.Contrast(img_night)
    img_night = enhancer_contrast.enhance(random.uniform(0.7, 0.9))
    return img_night


def make_rainy_noise_image(img):
    """
    画像を暗く・低コントラストにしたうえで、雨だれ風の白い線を多数描いて
    「雨天・低画質」っぽい見た目を作る。

    線の本数は画像面積の約1%。reCAPTCHAでよくある粗い画像を模している。
    """
    enhancer_brightness = ImageEnhance.Brightness(img)
    img_bad = enhancer_brightness.enhance(random.uniform(0.5, 0.8))
    enhancer_contrast = ImageEnhance.Contrast(img_bad)
    img_bad = enhancer_contrast.enhance(random.uniform(0.5, 0.8))

    draw = ImageDraw.Draw(img_bad)
    width, height = img.size
    num_drops = int((width * height) * 0.01)
    for _ in range(num_drops):
        x = random.randint(0, width)
        y = random.randint(0, height)
        length = random.randint(5, 20)
        draw.line(
            (x, y, x + random.randint(-2, 2), y + length),
            fill=(200, 200, 200),
            width=1,
        )
    return img_bad


def save_augmented_variants(img, save_dir, base_filename):
    """
    1枚のRGB画像から original / night / rain の3バリエーションを保存する。

    bus と other の両方がこの関数を通ることで、必ず同じ3点セットが作られ、
    クラス間で加工の偏りが生じないことを保証する。
    保存した枚数（常に3）を返す。
    """
    img = img.convert("RGB")

    # パターン1: 原本（昼・素の画像）
    img.save(os.path.join(save_dir, f"{base_filename}_original.jpg"))
    # パターン2: 夜
    make_night_image(img.copy()).save(
        os.path.join(save_dir, f"{base_filename}_night.jpg")
    )
    # パターン3: 雨・ノイズ
    make_rainy_noise_image(img.copy()).save(
        os.path.join(save_dir, f"{base_filename}_rain.jpg")
    )

    return 3
