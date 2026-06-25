import os
import json
import urllib.request
import zipfile
import random
from io import BytesIO
from PIL import Image

# bus と全く同じ加工を使うため、共通モジュールから読み込む
from data_augment import DEFAULT_SEED, save_augmented_variants

# --- 設定 ---
SAVE_DIR = "dataset/train/other"
MAX_BASE_IMAGES = 100  # 集めるベース画像の枚数（3パターン加工で合計300枚になります）

# 集めたい「バス以外」のカテゴリ設定
# 1. バスに似ているハードネガティブ（全体の約半分をこれにします）
SIMILAR_CLASSES = ['truck', 'car']
# 2. 明らかに違うオブジェクト
DIFFERENT_CLASSES = ['person', 'dog', 'cat', 'chair', 'couch', 'tv']

ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_FILE = "annotations_trainval2017.zip"
JSON_FILE = "annotations/instances_train2017.json"
# -----------


def main():
    # bus と同じシードで加工を再現可能にする
    random.seed(DEFAULT_SEED)

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

    # 1. 各カテゴリのIDを取得
    cat_name_to_id = {cat['name']: cat['id'] for cat in coco_data['categories']}
    bus_id = cat_name_to_id.get('bus')
    
    similar_ids = [cat_name_to_id[name] for name in SIMILAR_CLASSES if name in cat_name_to_id]
    different_ids = [cat_name_to_id[name] for name in DIFFERENT_CLASSES if name in cat_name_to_id]

    # 2. 【超重要】バスが写っている画像のIDをリストアップ（後で除外するため）
    bus_image_ids = set()
    for ann in coco_data['annotations']:
        if ann['category_id'] == bus_id:
            bus_image_ids.add(ann['image_id'])

    # 3. 指定したカテゴリの画像を集める
    print("条件に合う画像を検索中（バスが写り込んでいる画像は排除します）...")
    target_images = {'similar': set(), 'different': set()}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        
        # もしその画像にバスが1ミリでも写っていたら、絶対にスキップ！
        if img_id in bus_image_ids:
            continue
            
        if ann['category_id'] in similar_ids:
            target_images['similar'].add(img_id)
        elif ann['category_id'] in different_ids:
            target_images['different'].add(img_id)

    # リスト化してシャッフル
    similar_list = list(target_images['similar'])
    different_list = list(target_images['different'])
    random.shuffle(similar_list)
    random.shuffle(different_list)

    # 半分ずつ（50枚：50枚）選ぶ
    half_count = MAX_BASE_IMAGES // 2
    selected_image_ids = similar_list[:half_count] + different_list[:half_count]
    
    # 画像のURLを取得
    image_info = {img['id']: img for img in coco_data['images']}
    selected_images = [image_info[img_id] for img_id in selected_image_ids]
    random.shuffle(selected_images) # さらに全体を混ぜる

    print(f"\nダウンロードと加工を開始します（保存先: {SAVE_DIR}）...")
    processed_count = 0

    for img_meta in selected_images:
        image_url = img_meta['coco_url']
        base_filename = f"other_{img_meta['id']:012d}"

        try:
            req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                img_data = response.read()
            
            with Image.open(BytesIO(img_data)) as img:
                # bus と全く同じ加工（original / night / rain の3点）を適用する
                save_augmented_variants(img, SAVE_DIR, base_filename)

            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  ... ベース画像 {processed_count} 枚分の処理完了（合計 {processed_count * 3} 枚保存）")

        except Exception as e:
            print(f"画像 {image_url} の処理に失敗しました: {e}")

    print(f"\n完了しました！高品質な「バス以外（other）」の画像が '{SAVE_DIR}' に保存されました。")

if __name__ == "__main__":
    main()