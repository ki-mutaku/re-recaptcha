import os
from ultralytics import YOLO

def main():
    print("🚀 モデルを読み込んでいます...")
    model = YOLO('yolov8n.pt')

    # 1. 処理したいフォルダ名を設定
    folder_path = 'my_recaptcha_dataset'
    
    # フォルダの存在チェック
    if not os.path.exists(folder_path):
        print(f"⚠️ エラー: '{folder_path}' フォルダが見つかりません。作成して画像を入れてください。")
        return

    print(f"📁 フォルダ '{folder_path}' 内の画像を解析中...")
    
    # 2. YOLOにフォルダごと投げる（これだけで全自動で回してくれます）
    # 信頼度のしきい値（conf）を0.1（10%）まで下げ、
    # さらに結果の画像を自動保存（save=True）する設定にします
    results = model(folder_path, conf=0.1, save=True)

    # 3. 将来の「共起スコア」計算に使うための辞書（データボックス）
    # イメージ：{ "bus1.jpg": ["car", "bus", "person"], "bus2.jpg": ["traffic light", "car"] }
    dataset_scores = {}

    print("\n--- 📊 フォルダの一括処理結果 ---")
    
    for r in results:
        # 画像のファイル名を取得
        filename = os.path.basename(r.path)
        
        detected_labels = []
        for box in r.boxes:
            class_id = int(box.cls[0])
            label_name = model.names[class_id]
            detected_labels.append(label_name)
            
        # ファイル名と検出結果をセットにして保存
        dataset_scores[filename] = detected_labels
        
        # ターミナルに出力
        print(f" 📸 {filename}: {detected_labels}")


if __name__ == "__main__":
    main()