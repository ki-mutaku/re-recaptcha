import os
import json
import shutil
import glob
import random
import yaml

# =============================================
# 設定
# =============================================

# 入力：加工済み画像フォルダ（モザイクか霧、どちらか使いたい方を指定）
IMAGE_DIRS = {
    "mosaic": "test_images_mosaic",
    "fog":    "test_images_fog",
}

# 元画像のアノテーション（座標はどちらの加工でも共通）
ANNOTATION_JSON = "test_images_coco/annotations.json"

# 出力先
OUTPUT_DIR = "yolo_dataset"

# train/val の分割比率
TRAIN_RATIO = 0.8

# クラス定義（download_coco.pyで指定したクラスと合わせる）
CLASSES = ["bus", "car", "traffic light"]

# =============================================
# COCO → YOLO 座標変換
# =============================================

def coco_to_yolo(x, y, w, h, img_w, img_h):
    """
    COCO形式  : [x_left, y_top, width, height]（ピクセル値）
    YOLO形式  : [cx, cy, width, height]（0〜1の相対値、中心基準）
    """
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    rw = w / img_w
    rh = h / img_h
    # 念のため0〜1にクリップ
    return (
        round(min(max(cx, 0), 1), 6),
        round(min(max(cy, 0), 1), 6),
        round(min(max(rw, 0), 1), 6),
        round(min(max(rh, 0), 1), 6),
    )

# =============================================
# メイン処理
# =============================================

def main():
    # アノテーション読み込み
    if not os.path.exists(ANNOTATION_JSON):
        print(f"⚠️ {ANNOTATION_JSON} が見つかりません。download_coco.py を先に実行してください。")
        return

    with open(ANNOTATION_JSON) as f:
        annotations = json.load(f)

    class_to_id = {cls: i for i, cls in enumerate(CLASSES)}

    # 出力フォルダの作成
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

    total_images = 0
    total_labels = 0
    skipped = 0

    for source_name, image_dir in IMAGE_DIRS.items():
        if not os.path.exists(image_dir):
            print(f"⚠️ '{image_dir}' が見つかりません。スキップします。")
            continue

        image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        random.shuffle(image_paths)
        split_idx = int(len(image_paths) * TRAIN_RATIO)

        for i, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            stem = os.path.splitext(filename)[0]
            split = "train" if i < split_idx else "val"

            # アノテーションが存在するか確認
            if filename not in annotations or not annotations[filename]:
                skipped += 1
                continue

            # 画像サイズを取得（アノテーション内から逆算）
            # ※ bbox_pixel の x+w がimg_wを超えないことを利用して
            #   実際には cv2 で読む方が確実
            import cv2
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            img_h, img_w = img.shape[:2]

            # ラベルファイルを生成
            label_lines = []
            for box in annotations[filename]:
                label = box["label"]
                if label not in class_to_id:
                    continue  # 対象外クラスはスキップ
                class_id = class_to_id[label]
                x, y, w, h = box["bbox_pixel"]
                cx, cy, rw, rh = coco_to_yolo(x, y, w, h, img_w, img_h)
                label_lines.append(f"{class_id} {cx} {cy} {rw} {rh}")

            if not label_lines:
                skipped += 1
                continue

            # ファイル名にソース名をつけて衝突を防ぐ
            # 例: 000001.jpg → mosaic_000001.jpg
            new_filename = f"{source_name}_{filename}"
            new_stem = f"{source_name}_{stem}"

            # 画像コピー
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, "images", split, new_filename))

            # ラベル保存
            label_path = os.path.join(OUTPUT_DIR, "labels", split, f"{new_stem}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))

            total_images += 1
            total_labels += len(label_lines)

    # data.yaml の生成
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    yaml_content = {
        "path": os.path.abspath(OUTPUT_DIR),
        "train": "images/train",
        "val":   "images/val",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ 変換完了！")
    print(f"  🖼️  画像数  : {total_images} 枚")
    print(f"  🏷️  ラベル数: {total_labels} 件")
    print(f"  ⏭️  スキップ: {skipped} 枚（アノテーションなし）")
    print(f"\n📁 出力先: {OUTPUT_DIR}/")
    print(f"  ├─ images/train/  ├─ images/val/")
    print(f"  ├─ labels/train/  ├─ labels/val/")
    print(f"  └─ data.yaml  ← YOLOに渡すファイル")
    print(f"\n🚀 ファインチューニングはこのコマンドで実行できます：")
    print(f"  yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")

if __name__ == "__main__":
    main()