import os
import json
import urllib.request
import zipfile

# --- 設定 ---
SAVE_DIR = "dataset/train/bus"
MAX_IMAGES = 100      # ダウンロードする画像の最大枚数
MIN_AREA_RATIO = 0.1  # 【追加】バスの面積が画像全体の10%以上あるものだけを採用する（0.1 = 10%）
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_FILE = "annotations_trainval2017.zip"
JSON_FILE = "annotations/instances_train2017.json"
# -----------

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not os.path.exists(JSON_FILE):
        if not os.path.exists(ZIP_FILE):
            print("COCOアノテーションデータをダウンロードしています...")
            urllib.request.urlretrieve(ANNOTATION_URL, ZIP_FILE)
        print("Zipファイルを解凍しています...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")

    print("アノテーションデータを読み込んでいます...")
    with open(JSON_FILE, 'r') as f:
        coco_data = json.load(f)

    bus_category_id = None
    for cat in coco_data['categories']:
        if cat['name'] == 'bus':
            bus_category_id = cat['id']
            break

    # 画像IDから画像の幅と高さをすぐに引けるように辞書化
    image_info = {img['id']: img for img in coco_data['images']}

    print(f"画像全体の {MIN_AREA_RATIO*100}% 以上の大きさでバスが写っている画像を厳選しています...")
    bus_image_ids = set()
    
    for ann in coco_data['annotations']:
        if ann['category_id'] == bus_category_id:
            img_id = ann['image_id']
            
            if img_id in image_info:
                img_width = image_info[img_id]['width']
                img_height = image_info[img_id]['height']
                img_area = img_width * img_height
                
                # バスの面積を取得
                bus_area = ann['area']
                
                # 面積の割合を計算（バスの面積 ÷ 画像全体の面積）
                ratio = bus_area / img_area
                
                # 閾値（10%）以上なら採用リストに追加
                if ratio >= MIN_AREA_RATIO:
                    bus_image_ids.add(img_id)

    bus_images = [img for img in coco_data['images'] if img['id'] in bus_image_ids]
    print(f"条件をクリアしたバス画像が {len(bus_images)} 枚見つかりました。")

    download_count = 0
    for img in bus_images[:MAX_IMAGES]:
        image_url = img['coco_url']
        filename = os.path.join(SAVE_DIR, f"{img['id']:012d}.jpg")

        if not os.path.exists(filename):
            try:
                urllib.request.urlretrieve(image_url, filename)
                download_count += 1
                if download_count % 10 == 0:
                    print(f"  ... {download_count} 枚ダウンロード完了")
            except Exception as e:
                print(f"ダウンロード失敗: {e}")
        else:
            print(f"スキップ: {filename} は既に存在します。")

    print(f"\n完了しました！厳選された画像が {SAVE_DIR} に保存されています。")

if __name__ == "__main__":
    main()