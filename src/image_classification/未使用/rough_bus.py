import os
import cv2

#画像ファイルを指定して、粗くするコード

# 画像が入っているフォルダ
input_dir = "val2017"
# 粗くした画像を保存するフォルダ
output_dir = "bus"
# 保存先フォルダがなければ作る
os.makedirs(output_dir, exist_ok=True)
# 粗さの設定
# 数字を小さくするともっと粗くなる
rough_size = 130

for filename in os.listdir(input_dir):
    # jpg, jpeg, png だけ処理する
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 画像を読み込む
        img = cv2.imread(input_path)

        if img is None:
            print("読み込めませんでした:", filename)
            continue

        # 一度小さくする
        small = cv2.resize(img, (rough_size, rough_size))

        # 元の画像サイズに戻す
        height, width = img.shape[:2]
        rough_img = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )
        # 保存する
        cv2.imwrite(output_path, rough_img)

        print("保存しました:", output_path)

print("全部完了しました")