import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    # YOLOv8のセグメンテーションモデルをロード (必要に応じてパスを変更してください)
    model_path = 'yolov8n-seg.pt'
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # ディレクトリの設定。src直下のデータセット/生成ラベルを参照する。
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / 'my_recaptcha_dataset' / 'data'
    fog_dir = base_dir / 'test_images_fog'
    mosaic_dir = base_dir / 'test_images_mosaic'

    # ラベル出力用ディレクトリの作成
    label_fog_dir = base_dir / 'labels_fog'
    label_mosaic_dir = base_dir / 'labels_mosaic'
    label_fog_dir.mkdir(parents=True, exist_ok=True)
    label_mosaic_dir.mkdir(parents=True, exist_ok=True)

    # 学習用のアノテーション(.txt)を保存する処理
    image_files = list(data_dir.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images in {data_dir}")
    
    for img_path in image_files:
        # 元画像で推論を実行
        results = model(img_path, verbose=False)
        
        for result in results:
            label_lines = []
            
            # マスクが検出された場合
            if result.masks is not None:
                # xynは 0.0〜1.0 に正規化されたポリゴンの座標
                segments = result.masks.xyn
                classes = result.boxes.cls.cpu().numpy()
                
                for cls, segment in zip(classes, segments):
                    # YOLO format: class x1 y1 x2 y2 ...
                    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in segment])
                    line = f"{int(cls)} {coords}"
                    label_lines.append(line)
            
            label_content = "\n".join(label_lines)
            
            # 1. ぼかし画像用のラベルを別ディレクトリに保存
            fog_img_path = fog_dir / img_path.name
            if fog_img_path.exists():
                label_fog_path = label_fog_dir / img_path.with_suffix('.txt').name
                with open(label_fog_path, 'w') as f:
                    if label_content:
                        f.write(label_content + '\n')
            
            # 2. モザイク画像用のラベルを別ディレクトリに保存
            mosaic_img_path = mosaic_dir / img_path.name
            if mosaic_img_path.exists():
                label_mosaic_path = label_mosaic_dir / img_path.with_suffix('.txt').name
                with open(label_mosaic_path, 'w') as f:
                    if label_content:
                        f.write(label_content + '\n')
                        
        print(f"Processed: {img_path.name}")

    print("完了しました。labels_fog と labels_mosaic ディレクトリに YOLOv8学習用のラベル(.txt)を生成しました。")

if __name__ == '__main__':
    main()
