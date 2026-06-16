import cv2
import os
import glob
import numpy as np

def apply_mosaic(image, ratio=0.1):
    """画像にモザイク処理をかける"""
    h, w = image.shape[:2]
    # 一旦小さく縮小して、元のサイズに拡大することでモザイク化
    small = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_fog(image, blur_level=15):
    """画像に霧（ぼかし）処理をかける"""
    # ガウシアンブラーで霧のような効果を出す（数値が大きいほど濃い霧）
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)

def main():
    input_dir = 'my_recaptcha_dataset/data'
    output_dir_mosaic = 'test_images_mosaic'
    output_dir_fog = 'test_images_fog'

    os.makedirs(output_dir_mosaic, exist_ok=True)
    os.makedirs(output_dir_fog, exist_ok=True)

    # フォルダ内の全jpg画像を取得
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    
    if not image_paths:
        print(f"⚠️ '{input_dir}' に画像がありません。")
        return

    print(f"🎨 {len(image_paths)}枚の画像にモザイクと霧をかけています...")

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)

        if img is None:
            continue

        # モザイク処理して保存
        mosaic_img = apply_mosaic(img, ratio=0.08) # 0.08はモザイクの粗さ（下げるほど粗い）
        cv2.imwrite(os.path.join(output_dir_mosaic, filename), mosaic_img)

        # 霧処理して保存
        fog_img = apply_fog(img, blur_level=25) # 25は霧の濃さ（奇数にする必要あり）
        cv2.imwrite(os.path.join(output_dir_fog, filename), fog_img)

    print(f"✅ 完了！ '{output_dir_mosaic}' と '{output_dir_fog}' フォルダを確認してください。")

if __name__ == "__main__":
    main()