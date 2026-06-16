import cv2
import os
import glob

def apply_clahe(image, clip_limit=3.0, tile_size=(8, 8)):
    """
    CLAHEを使って画像を鮮明にする関数
    """
    # 【超重要ポイント】
    # BGRのまま平坦化すると色がバグるため、一度「LAB色空間」に変換します。
    # L(明るさ)、A(赤〜緑)、B(黄〜青) に分離します。
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # CLAHE（適応的ヒストグラム平坦化）の準備
    # clipLimitが大きいほどコントラストが強くなります（通常は2.0〜4.0がおすすめ）
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # Lチャンネル（明るさのデータ）に対してのみ平坦化を実行
    cl = clahe.apply(l_channel)

    # 平坦化した明るさ(cl)と、元の色(a, b)を再結合
    merged_lab = cv2.merge((cl, a_channel, b_channel))

    # BGR（元の画像形式）に戻す
    recovered_img = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return recovered_img

def main():
    # 読み込むフォルダ（前回作った霧の画像）
    input_dir = 'test_images_fog'
    # 保存するフォルダ
    output_dir = 'test_images_recovered'

    os.makedirs(output_dir, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg'))

    if not image_paths:
        print(f"⚠️ '{input_dir}' に画像がありません。")
        return

    print(f"🪄 {len(image_paths)}枚の画像の霧を晴らして、AIの視力を回復させています...")

    for path in image_paths:
        filename = os.path.basename(path)
        img = cv2.imread(path)

        if img is None:
            continue

        # CLAHE関数を適用
        recovered_img = apply_clahe(img, clip_limit=3.0)

        # 処理後の画像を保存
        cv2.imwrite(os.path.join(output_dir, filename), recovered_img)

    print(f"✅ 完了！ '{output_dir}' フォルダを開いて、画質がどう変化したか確認してください。")

if __name__ == "__main__":
    main()