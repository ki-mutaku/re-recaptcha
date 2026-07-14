import os
import json
import urllib.request
import zipfile
import random
from io import BytesIO
from PIL import Image, ImageEnhance, ImageDraw

# --- 設定 ---
SAVE_DIR = "dataset/train/bus"
MAX_NEW_IMAGES = 100  # 新しく追加したいベース画像の枚数
MIN_AREA_RATIO = 0.1  # バスの面積が画像全体の10%以上

ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_FILE = "annotations_trainval2017.zip"
JSON_FILE = "annotations/instances_train2017.json"
# -----------

def make_night_image(img):
    """夜の画像にする"""
    enhancer_brightness = ImageEnhance.Brightness(img)
    img_night = enhancer_brightness.enhance(random.uniform(0.2, 0.4))
    enhancer_contrast = ImageEnhance.Contrast(img_night)
    img_night = enhancer_contrast.enhance(random.uniform(0.7, 0.9))
    return img_night

def make_rainy_noise_image(img):
    """雨風の悪天候/低画質ノイズ画像にする"""
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
        draw.line((x, y, x + random.randint(-2, 2), y + length), fill=(200, 200, 200), width=1)
        
    return img_bad

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. すでに保存されている画像のIDをチェック（ダブり防止機能）
    existing_ids = set()
    for filename in os.listdir(SAVE_DIR):
        if filename.endswith("_original.jpg"):
            try:
                # ファイル名（例: 000000123456_original.jpg）からID部分を抽出
                img_id = int(filename.split('_')[0])
                existing_ids.add(img_id)
            except ValueError:
                pass
    
    print(f"フォルダ内をチェックしました: すでに {len(existing_ids)} 枚のベース画像が存在します（これらは除外します）。")

    # 2. アノテーションデータの準備
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

    bus_category_id = next((cat['id'] for cat in coco_data['categories'] if cat['name'] == 'bus'), None)
    image_info = {img['id']: img for img in coco_data['images']}

    # 3. 条件に合う「新しい」バス画像だけを厳選
    bus_image_ids = set()
    for ann in coco_data['annotations']:
        if ann['category_id'] == bus_category_id:
            img_id = ann['image_id']
            # すでに持っている画像IDはここで弾く
            if img_id in existing_ids:
                continue
                
            if img_id in image_info:
                ratio = ann['area'] / (image_info[img_id]['width'] * image_info[img_id]['height'])
                if ratio >= MIN_AREA_RATIO:
                    bus_image_ids.add(img_id)

    # 対象となる新しい画像をリスト化してシャッフル
    new_bus_images = [img for img in coco_data['images'] if img['id'] in bus_image_ids]
    random.shuffle(new_bus_images) 

    print(f"新しく追加可能なバス画像が {len(new_bus_images)} 枚見つかりました。")
    if len(new_bus_images) == 0:
        print("追加できる新しい画像がありません。面積条件（MIN_AREA_RATIO）を緩めてみてください。")
        return

    print(f"\n完全に新しい {MAX_NEW_IMAGES} 枚のダウンロードと加工を開始します...")
    processed_count = 0

    # 4. ダウンロード ＆ 加工 ＆ 保存
    for img_meta in new_bus_images:
        if processed_count >= MAX_NEW_IMAGES:
            break

        image_url = img_meta['coco_url']
        base_filename = f"{img_meta['id']:012d}"

        try:
            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_data = response.read()
            
            with Image.open(BytesIO(img_data)) as img:
                img = img.convert('RGB')
                
                # パターン1: 原本
                img.save(os.path.join(SAVE_DIR, f"{base_filename}_original.jpg"))
                # パターン2: 夜
                img_night = make_night_image(img.copy())
                img_night.save(os.path.join(SAVE_DIR, f"{base_filename}_night.jpg"))
                # パターン3: 雨・ノイズ
                img_rain = make_rainy_noise_image(img.copy())
                img_rain.save(os.path.join(SAVE_DIR, f"{base_filename}_rain.jpg"))

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  ... 新規ベース画像 {processed_count} 枚分の処理完了（合計 {processed_count * 3} 枚追加）")

        except Exception as e:
            print(f"画像 {image_url} の処理に失敗しました: {e}")

    print(f"\n完了しました！ダブりのない新しいバス画像が '{SAVE_DIR}' に追加されました。")

if __name__ == "__main__":
    main()