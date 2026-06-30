import os
import json
import random
import urllib.request
import zipfile
from io import BytesIO

from PIL import Image

from data_augment import DEFAULT_SEED, clean_stale_variants, save_augmented_variants

# --- 設定 ---
SAVE_DIR = "dataset/train/bus"
MAX_BASE_IMAGES = 100  # 集めるベース画像の枚数（3パターン加工で合計300枚になります）
MIN_AREA_RATIO = 0.1  # バスの面積が画像全体の10%以上あるものだけを採用する（0.1 = 10%）
ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ZIP_FILE = "annotations_trainval2017.zip"
JSON_FILE = "annotations/instances_train2017.json"
# -----------


def main():
    # other と同じシードで加工を再現可能にする
    random.seed(DEFAULT_SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 旧フィルタの生成物（例: _night.jpg/_rain.jpg）が残っていると、現行フィルタの
    # 生成分と混在して学習データが汚れるので、現行サフィックス以外は実行前に削除する
    removed = clean_stale_variants(SAVE_DIR)
    if removed:
        print(f"【清掃】{SAVE_DIR} の旧フィルタ生成物を {removed} 枚削除しました。")

    if not os.path.exists(JSON_FILE):
        if not os.path.exists(ZIP_FILE):
            print("COCOアノテーションデータをダウンロードしています...")
            urllib.request.urlretrieve(ANNOTATION_URL, ZIP_FILE)
        print("Zipファイルを解凍しています...")
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")

    print("アノテーションデータを読み込んでいます...")
    with open(JSON_FILE, "r") as f:
        coco_data = json.load(f)

    bus_category_id = None
    for cat in coco_data["categories"]:
        if cat["name"] == "bus":
            bus_category_id = cat["id"]
            break

    # 画像IDから画像の幅と高さをすぐに引けるように辞書化
    image_info = {img["id"]: img for img in coco_data["images"]}

    print(
        f"画像全体の {MIN_AREA_RATIO * 100}% 以上の大きさでバスが写っている画像を厳選しています..."
    )
    bus_image_ids = set()

    for ann in coco_data["annotations"]:
        if ann["category_id"] == bus_category_id:
            img_id = ann["image_id"]

            if img_id in image_info:
                img_width = image_info[img_id]["width"]
                img_height = image_info[img_id]["height"]
                img_area = img_width * img_height

                # バスの面積を取得
                bus_area = ann["area"]

                # 面積の割合を計算（バスの面積 ÷ 画像全体の面積）
                ratio = bus_area / img_area

                # 閾値（10%）以上なら採用リストに追加
                if ratio >= MIN_AREA_RATIO:
                    bus_image_ids.add(img_id)

    bus_images = [img for img in coco_data["images"] if img["id"] in bus_image_ids]
    print(f"条件をクリアしたバス画像が {len(bus_images)} 枚見つかりました。")

    print(f"\nダウンロードと加工を開始します（保存先: {SAVE_DIR}）...")
    processed_count = 0

    for img_meta in bus_images[:MAX_BASE_IMAGES]:
        image_url = img_meta["coco_url"]
        base_filename = f"bus_{img_meta['id']:012d}"

        # 3種すべて既に存在するならスキップ（再実行を速くする）
        variants = [f"{base_filename}_original.jpg", f"{base_filename}_degraded1.jpg", f"{base_filename}_degraded2.jpg"]
        if all(os.path.exists(os.path.join(SAVE_DIR, v)) for v in variants):
            continue

        try:
            req = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                img_data = response.read()

            with Image.open(BytesIO(img_data)) as img:
                # other と全く同じ加工（original / night / rain の3点）を適用する
                save_augmented_variants(img, SAVE_DIR, base_filename)

            processed_count += 1
            if processed_count % 10 == 0:
                print(
                    f"  ... ベース画像 {processed_count} 枚分の処理完了（合計 {processed_count * 3} 枚保存）"
                )

        except Exception as e:
            print(f"画像 {image_url} の処理に失敗しました: {e}")

    print(f"\n完了しました！厳選・加工されたバス画像が {SAVE_DIR} に保存されています。")


if __name__ == "__main__":
    main()
