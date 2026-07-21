import cv2
from pathlib import Path

def apply_mosaic(image, ratio=0.1):
    """縮小後に最近傍補間で拡大し、低解像度風のモザイク画像を作る。"""
    h, w = image.shape[:2]
    small_size = (int(w * ratio), int(h * ratio))
    small = cv2.resize(image, small_size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_fog(image, blur_level=15):
    """ガウシアンブラーを適用し、霧で輪郭がぼけた画像を作る。"""
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)


def main():
    """src直下の元画像から、同名のモザイク画像と霧画像を生成する。"""
    base_dir = Path(__file__).resolve().parents[1]
    input_dir = base_dir / "my_recaptcha_dataset" / "data"
    output_dir_mosaic = base_dir / "test_images_mosaic"
    output_dir_fog = base_dir / "test_images_fog"

    output_dir_mosaic.mkdir(parents=True, exist_ok=True)
    output_dir_fog.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(input_dir.glob("*.jpg"))
    
    if not image_paths:
        print(f"⚠️ '{input_dir}' に画像がありません。")
        return

    print(f"🎨 {len(image_paths)}枚の画像にモザイクと霧をかけています...")

    for path in image_paths:
        img = cv2.imread(str(path))

        if img is None:
            continue

        mosaic_img = apply_mosaic(img, ratio=0.08)
        cv2.imwrite(str(output_dir_mosaic / path.name), mosaic_img)

        fog_img = apply_fog(img, blur_level=25)
        cv2.imwrite(str(output_dir_fog / path.name), fog_img)

    print(f"✅ 完了！ '{output_dir_mosaic}' と '{output_dir_fog}' フォルダを確認してください。")

if __name__ == "__main__":
    main()
