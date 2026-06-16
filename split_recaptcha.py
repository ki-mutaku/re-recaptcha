import os
from PIL import Image

# --- 設定 ---
# 1. 保存したreCAPTCHA画像（マス目部分のみ）
INPUT_IMAGE = "sample.jpg"

# 2. 分割した画像の保存先フォルダ
OUTPUT_DIR = "test_images"
# -----------

def main():
    if not os.path.exists(INPUT_IMAGE):
        print(f"エラー: '{INPUT_IMAGE}' が見つかりません。")
        print("画像を保存し、ファイル名が正しいか確認してください。")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        with Image.open(INPUT_IMAGE) as img:
            # 画像の幅と高さを取得
            width, height = img.size
            
            # 3等分するための幅と高さを計算
            tile_w = width // 3
            tile_h = height // 3
            
            print(f"元の画像サイズ: {width}x{height} -> 1マス: {tile_w}x{tile_h}")
            
            count = 0
            # 縦に3行、横に3列のループを回して切り抜く
            for row in range(3):
                for col in range(3):
                    # 切り抜く座標 (左, 上, 右, 下)
                    left = col * tile_w
                    upper = row * tile_h
                    right = left + tile_w
                    lower = upper + tile_h
                    
                    # 画像を切り抜き (Crop)
                    tile = img.crop((left, upper, right, lower))
                    
                    # ファイル名を tile_0.jpg 〜 tile_8.jpg として保存
                    save_path = os.path.join(OUTPUT_DIR, f"tile_{count}.jpg")
                    tile.save(save_path)
                    
                    print(f"  -> '{save_path}' を保存しました。")
                    count += 1
                    
        print(f"\n分割完了！ 9枚のテスト画像を '{OUTPUT_DIR}' フォルダに準備しました。")
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()