import argparse
import random
from pathlib import Path

import cv2


DEFAULT_SEED = 42
DEFAULT_MOSAIC_RATIO_MIN = 0.05
DEFAULT_MOSAIC_RATIO_MAX = 0.10
DEFAULT_FOG_BLUR_MIN = 15
DEFAULT_FOG_BLUR_MAX = 25


def apply_mosaic(image, ratio=0.1):
    """縮小後に最近傍補間で拡大し、低解像度風のモザイク画像を作る。"""
    h, w = image.shape[:2]
    small_size = (int(w * ratio), int(h * ratio))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_fog(image, blur_level=15):
    """ガウシアンブラーを適用し、霧で輪郭がぼけた画像を作る。"""
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)


def random_odd_integer(rng, minimum, maximum):
    """指定範囲からGaussianBlur用の奇数を再現可能に選ぶ。"""
    odd_values = list(range(minimum | 1, maximum + 1, 2))
    if not odd_values:
        raise ValueError("霧のぼかし強度には奇数を1つ以上指定してください。")
    return rng.choice(odd_values)


def parse_args():
    """ノイズ強度の範囲と乱数シードをCLIから受け取る。"""
    parser = argparse.ArgumentParser(
        description="画像ごとに強度を変えたfog/mosaic画像を生成します。"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--mosaic-ratio-min",
        type=float,
        default=DEFAULT_MOSAIC_RATIO_MIN,
    )
    parser.add_argument(
        "--mosaic-ratio-max",
        type=float,
        default=DEFAULT_MOSAIC_RATIO_MAX,
    )
    parser.add_argument(
        "--fog-blur-min",
        type=int,
        default=DEFAULT_FOG_BLUR_MIN,
    )
    parser.add_argument(
        "--fog-blur-max",
        type=int,
        default=DEFAULT_FOG_BLUR_MAX,
    )
    return parser.parse_args()


def main():
    """src直下の元画像から、同名のモザイク画像と霧画像を生成する。"""
    args = parse_args()
    if not 0 < args.mosaic_ratio_min <= args.mosaic_ratio_max <= 1:
        raise ValueError(
            "モザイク比率は0より大きく1以下の範囲で指定してください。"
        )
    if not 1 <= args.fog_blur_min <= args.fog_blur_max:
        raise ValueError("霧のぼかし強度の範囲が不正です。")

    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "my_recaptcha_dataset" / "data"
    output_dir_mosaic = base_dir / "test_images_mosaic"
    output_dir_fog = base_dir / "test_images_fog"

    output_dir_mosaic.mkdir(parents=True, exist_ok=True)
    output_dir_fog.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))
    rng = random.Random(args.seed)

    if not image_paths:
        print(f"⚠️ '{input_dir}' に画像がありません。")
        return

    print(f"🎨 {len(image_paths)}枚の画像にモザイクと霧をかけています...")

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            continue

        mosaic_ratio = rng.uniform(
            args.mosaic_ratio_min,
            args.mosaic_ratio_max,
        )
        mosaic_img = apply_mosaic(img, ratio=mosaic_ratio)
        cv2.imwrite(str(output_dir_mosaic / path.name), mosaic_img)

        fog_blur_level = random_odd_integer(
            rng,
            args.fog_blur_min,
            args.fog_blur_max,
        )
        fog_img = apply_fog(img, blur_level=fog_blur_level)
        cv2.imwrite(str(output_dir_fog / path.name), fog_img)

    print(f"✅ 完了！ '{output_dir_mosaic}' と '{output_dir_fog}' フォルダを確認してください。")

if __name__ == "__main__":
    main()
