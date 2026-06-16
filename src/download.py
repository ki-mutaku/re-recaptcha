import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil
import json
import csv

def main():
    print("🚀 COCOデータセットから特定の画像を検索・ダウンロードします...")

    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["bus", "car", "traffic light"],
        max_samples=30,
    )

    export_dir = "test_images_coco"
    os.makedirs(export_dir, exist_ok=True)

    print(f"\n📂 '{export_dir}' フォルダに画像をまとめています...")

    annotations = {}   # JSON用（後続スクリプトで流用）
    csv_rows = []      # CSV用（目視確認用）

    for sample in dataset:
        original_path = sample.filepath
        filename = os.path.basename(original_path)
        shutil.copy(original_path, os.path.join(export_dir, filename))

        # 画像サイズを取得（metadataが空の場合は手動で読む）
        if sample.metadata and sample.metadata.width:
            img_w = sample.metadata.width
            img_h = sample.metadata.height
        else:
            import cv2
            img = cv2.imread(original_path)
            img_h, img_w = img.shape[:2]

        boxes = []
        if sample.ground_truth:
            for det in sample.ground_truth.detections:
                # FiftyOneの座標は [x_left, y_top, width, height] で 0〜1 の相対値
                rx, ry, rw, rh = det.bounding_box

                # ピクセル座標に変換
                px = round(rx * img_w, 1)
                py = round(ry * img_h, 1)
                pw = round(rw * img_w, 1)
                ph = round(rh * img_h, 1)

                boxes.append({
                    "label": det.label,
                    "bbox_pixel": [px, py, pw, ph]  # [x, y, w, h]
                })

                # CSV用の行を追加
                csv_rows.append({
                    "filename": filename,
                    "label": det.label,
                    "x": px,
                    "y": py,
                    "w": pw,
                    "h": ph,
                    "x2": round(px + pw, 1),  # 右下X（確認しやすいよう追加）
                    "y2": round(py + ph, 1),  # 右下Y（確認しやすいよう追加）
                    "img_w": img_w,
                    "img_h": img_h,
                })

        annotations[filename] = boxes

    # --- JSON保存（後続スクリプト用） ---
    json_path = os.path.join(export_dir, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # --- CSV保存（目視確認用） ---
    csv_path = os.path.join(export_dir, "annotations_preview.csv")
    fieldnames = ["filename", "label", "x", "y", "w", "h", "x2", "y2", "img_w", "img_h"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n✅ 完了！ '{export_dir}' フォルダを確認してください。")
    print(f"  📄 annotations.json        → 後続スクリプト用")
    print(f"  📊 annotations_preview.csv → 座標の目視確認用")
    print(f"  🖼️  検出ボックス数: {len(csv_rows)} 件 / {len(annotations)} 枚")

if __name__ == "__main__":
    main()