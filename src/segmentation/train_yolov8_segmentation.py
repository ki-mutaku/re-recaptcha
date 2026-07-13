import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

def setup_dataset():
    # src直下の生成データを基準に、セグメンテーション用データセットを組み立てる。
    # このファイルは src/segmentation/ 配下にあるため、parents[1] が src ディレクトリになる。
    base_dir = Path(__file__).resolve().parents[1]
    
    # 元データのパス（画像と生成したラベル）
    src_dirs = [
        {"images": base_dir / 'test_images_fog', "labels": base_dir / 'labels_fog'},
        {"images": base_dir / 'test_images_mosaic', "labels": base_dir / 'labels_mosaic'}
    ]
    
    # YOLOv8用のデータセット作成先ディレクトリ
    dataset_dir = base_dir / 'yolo_dataset'
    train_images_dir = dataset_dir / 'images' / 'train'
    val_images_dir = dataset_dir / 'images' / 'val'
    train_labels_dir = dataset_dir / 'labels' / 'train'
    val_labels_dir = dataset_dir / 'labels' / 'val'
    
    # ディレクトリを作成（すでにある場合は上書きしないよう注意するか、再作成する）
    for d in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    all_data = []
    
    print("データセットを収集しています...")
    # 各ソースディレクトリから対応する画像とラベルを収集
    for src in src_dirs:
        img_dir = src["images"]
        lbl_dir = src["labels"]
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"警告: {img_dir} または {lbl_dir} が見つかりません。スキップします。")
            continue
            
        # 画像ファイルを取得し、対応するラベルがあるか確認
        for img_path in img_dir.glob('*.jpg'):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                # 画像のパスとラベルのパスをペアで保存
                all_data.append((img_path, lbl_path))

    if not all_data:
        raise ValueError("有効な画像とラベルのペアが見つかりませんでした。")
        
    print(f"合計 {len(all_data)} 件のデータペアが見つかりました。")
    
    # データをシャッフルして 8:2 (学習:検証) に分割
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"学習用: {len(train_data)} 件, 検証用: {len(val_data)} 件に分割してコピーします...")
    
    def copy_data(data_list, img_dest, lbl_dest):
        for img_path, lbl_path in data_list:
            # フォルダ名による名前の衝突を避けるため、プレフィックスをつける(任意)
            prefix = img_path.parent.name.replace('test_images_', '') + "_"
            new_img_name = prefix + img_path.name
            new_lbl_name = prefix + lbl_path.name
            
            shutil.copy2(img_path, img_dest / new_img_name)
            shutil.copy2(lbl_path, lbl_dest / new_lbl_name)
            
    copy_data(train_data, train_images_dir, train_labels_dir)
    copy_data(val_data, val_images_dir, val_labels_dir)
    
    return dataset_dir

def create_yaml(dataset_dir, model):
    yaml_path = dataset_dir / 'data.yaml'
    
    # YOLOv8モデル(COCO事前学習済み)が持っているクラス名を取得
    class_names = model.names 
    
    yaml_data = {
        'train': str(dataset_dir / 'images' / 'train'),
        'val': str(dataset_dir / 'images' / 'val'),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
    print(f"{yaml_path} を作成しました。")
    return yaml_path

def main():
    # 1. データの準備と分割
    dataset_dir = setup_dataset()
    
    # 2. モデルの読み込み (COCOのクラス名を取得するためにも先読み)
    print("モデル yolov8n-seg.pt をロードしています...")
    model = YOLO('yolov8n-seg.pt')
    
    # 3. data.yaml の生成
    yaml_path = create_yaml(dataset_dir, model)
    
    # 4. ファインチューニングの実行
    print("ファインチューニングを開始します...")
    model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,          # お使いのPCのスペックに合わせて調整してください(8や32など)
        device='mps',      # Mac(M1/M2等)の場合は 'mps' が推奨。Intel Macなら 'cpu'
        project='yolov8_segmentation_runs',
        name='fog_mosaic_finetune'
    )
    
    print("学習が完了しました！")

if __name__ == '__main__':
    main()
