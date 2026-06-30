"""
データ加工（本物reCAPTCHA寄りの劣化）の共通モジュール。

download_train_bus.py と download_train_other.py の両方から読み込んで使う。
bus クラスと other クラスに「全く同じ」加工を適用することで、
「劣化した画像 ＝ other」という偏った学習を防ぐのが目的。
（bus を素の昼間画像だけにすると、劣化したバスを一度も学習できず誤判定する）

make_night_image・make_rainy_noise_image は旧フィルタ（暗く＋白線）。
本物とのFID比較で逆効果と判定済みだが、比較用に eval/filter_validity.py から
参照されるため残している。学習データ生成（save_augmented_variants）では
make_recaptcha_like_image（新フィルタ）のみを使う。
"""

import io
import os
import random

from PIL import Image, ImageEnhance, ImageDraw, ImageFilter

# 加工結果を再現可能にするための乱数シード。
# bus と other で同じ値を使うことで、毎回同じ条件のデータセットを作れる。
DEFAULT_SEED = 42

# save_augmented_variants が生成するファイル名の末尾。
# フィルタ変更時の混在チェック（clean_stale_variants）や train/val分割
# （split_train_val.py）で「どれが現行フィルタの生成物か」を判定する基準として使う。
VARIANT_SUFFIXES = ("_original.jpg", "_degraded1.jpg", "_degraded2.jpg")


def clean_stale_variants(save_dir):
    """
    save_dir 内にある、現行の VARIANT_SUFFIXES に一致しないファイルを削除する。

    フィルタ（make_recaptcha_like_image など）を変更した後に
    download_train_bus.py / download_train_other.py を再実行すると、
    古いフィルタのファイル（例: 旧フィルタの "_night.jpg"/"_rain.jpg"）が
    削除されずに残り、新しい生成分と混在したまま学習されてしまう。
    生成前にこれを呼んで一掃することで、常に単一フィルタのデータセットを保つ。
    """
    if not os.path.isdir(save_dir):
        return 0
    removed = 0
    for fname in os.listdir(save_dir):
        path = os.path.join(save_dir, fname)
        if os.path.isfile(path) and fname.lower().endswith((".jpg", ".jpeg", ".png")) and not fname.endswith(VARIANT_SUFFIXES):
            os.remove(path)
            removed += 1
    return removed


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


def make_recaptcha_like_image(img):
    """
    本物の reCAPTCHA 画像に寄せた劣化を合成する。

    本物（nobodyPerfecZ/recaptchav2-29k）の bus 画像を観察すると、劣化は
      - 100x100 低解像度由来のボケ・細部の潰れ
      - JPEG ブロックノイズ
      - 彩度・コントラストの低下
    が主で、夜の暗さや雨だれの線は無い。よって make_night/make_rain とは別物として、
    「縮小して戻す＋軽いボケ＋彩度低下＋JPEG再圧縮」で本物寄りの劣化を作る。

    パラメータはランダム幅を持たせ、毎回少しずつ違う劣化を生成する。
    """
    img = img.convert("RGB")
    w, h = img.size

    # 1) 低解像度化: いったん小さくして元サイズへ戻し、細部を潰す
    scale = random.uniform(0.3, 0.6)
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    img = img.resize((small_w, small_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)

    # 2) 軽いガウシアンぼかし
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))

    # 3) 彩度・コントラストを少し落とす
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.0))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.0))

    # 4) JPEG 再圧縮（ブロックノイズ）— 本物のノイズ再現に一番効く
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=random.randint(20, 40))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_augmented_variants(img, save_dir, base_filename):
    """
    1枚のRGB画像から original / degraded1 / degraded2 の3バリエーションを保存する。

    degraded1・degraded2 は make_recaptcha_like_image（低解像度＋ぼけ＋JPEG再圧縮）を
    2回適用したもの。乱数幅があるため毎回違う劣化になる。
    本物reCAPTCHA（nobodyPerfecZ/recaptchav2-29k）とのFID比較で、旧フィルタ
    （make_night_image/make_rainy_noise_image）より本物に近いと確認済み
    （画像分類/eval/filter_validity.py の結果）。

    bus と other の両方がこの関数を通ることで、必ず同じ3点セットが作られ、
    クラス間で加工の偏りが生じないことを保証する。
    保存した枚数（常に3）を返す。
    """
    img = img.convert("RGB")

    # パターン1: 原本（昼・素の画像）
    img.save(os.path.join(save_dir, f"{base_filename}_original.jpg"))
    # パターン2・3: 本物reCAPTCHA寄りの劣化（乱数で2バリエーション）
    make_recaptcha_like_image(img.copy()).save(
        os.path.join(save_dir, f"{base_filename}_degraded1.jpg")
    )
    make_recaptcha_like_image(img.copy()).save(
        os.path.join(save_dir, f"{base_filename}_degraded2.jpg")
    )

    return 3
