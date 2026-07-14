import cv2
import os

def check_overlap(tile_box, target_boxes):
    """
    切り出したタイルの中に、指定されたお題（車など）が写っているかを判定する機能。
    タイルの座標と、お題のバウンディングボックスの座標が少しでも重なっていれば、
    「このタイルにはお題が写っている」とみなし True(1) を返す。
    
    引数:
        tile_box: (x_min, y_min, x_max, y_max) タイルの左上と右下の座標
        target_boxes: リスト [(x_min, y_min, x_max, y_max), ...] 複数の車の座標など
    """
    tx_min, ty_min, tx_max, ty_max = tile_box
    
    for box in target_boxes:
        bx_min, by_min, bx_max, by_max = box
        
        # タイルと対象物の四角形が重なっているか（交差判定）を行う計算。
        # 左右・上下の境界線が互いにクロスしているかで判定する。
        if (tx_max > bx_min and tx_min < bx_max and 
            ty_max > by_min and ty_min < by_max):
            return True # 1つでも重なっていれば正解タイル
            
    return False # どの対象物とも重なっていなければハズレタイル

def slice_and_label_image(image_path, target_boxes, grid_size=(3, 3)):
    """
    入力画像を読み込み、指定されたグリッド（今回は3x3）に切り分ける機能。
    同時に、各タイルに「お題が含まれているか」の判定を行い、
    結果を「タイル画像」と「0/1のラベル」のセットとして出力する。
    """
    # 1. 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        print("画像が見つかりません！パスを確認してください。")
        return []

    # 画像の高さ(h)と幅(w)を取得
    h, w, _ = img.shape
    
    # 1マスの幅と高さを計算する（割り切れない場合は切り捨て）
    tile_w = w // grid_size[1]
    tile_h = h // grid_size[0]
    
    results = [] # ここに9枚のタイル画像とラベルを格納する
    tile_index = 0
    
    # 2. 縦横のループを回して3x3に分割する
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # 現在のタイルの左上と右下の座標を計算する
            x_min = col * tile_w
            y_min = row * tile_h
            x_max = x_min + tile_w
            y_max = y_min + tile_h
            
            # 画像データから、計算した座標の範囲だけを配列のスライスで切り抜く
            tile_img = img[y_min:y_max, x_min:x_max]
            
            # 3. このタイルにお題（車など）が含まれているか判定する
            tile_box = (x_min, y_min, x_max, y_max)
            has_target = check_overlap(tile_box, target_boxes)
            
            # 正解なら 1、ハズレなら 0 をラベルとする
            label = 1 if has_target else 0
            
            # 結果を保存
            results.append({
                "index": tile_index,
                "image": tile_img,
                "label": label
            })
            
            tile_index += 1
            
    return results

# ==========================================
# 動作確認用テスト
# ==========================================
if __name__ == "__main__":
    # テスト用の画像パス（ご自身のPCにある適当な画像の名前を入れてください）
    TEST_IMAGE_PATH = "sample.jpg" 
    
    # 仮想の「車の座標データ（バウンディングボックス）」を定義
    # (x_min, y_min, x_max, y_max) の形式。※画像サイズに合わせて適当に変えてみてください
    dummy_car_boxes = [
        (100, 100, 200, 200) 
    ]
    
    # スライスとラベル付けを実行
    tiles_data = slice_and_label_image(TEST_IMAGE_PATH, dummy_car_boxes)
    
    # 結果の確認
    if tiles_data:
        print("=== スライス結果 ===")
        for data in tiles_data:
            print(f"タイル {data['index']}: ラベル {data['label']} (サイズ: {data['image'].shape})")