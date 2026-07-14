from ultralytics import YOLO
import os

def main():
    print("🚀 進化したオリジナルAIモデルを読み込んでいます...")
    
    # 1. あなたが育てたモデルのパスを指定します
    model_path = '/Users/yooseiisikawa/sakuhin/runs/detect/train-2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"⚠️ エラー: モデルが見つかりません。パス({model_path})が正しいか確認してください。")
        return
        
    model = YOLO(model_path)

    # 2. テストしたいフォルダを指定（まずは霧の画像で実力試し！）
    folder_path = 'test_images_coco'

    print(f"📸 '{folder_path}' の画像でテストを実施します...")

    # 3. 推論実行
    # save=True にすることで、答え合わせの枠付き画像が自動保存されます
    # conf=0.25 (25%以上の自信があるものだけ表示)
    results = model(folder_path, conf=0.25, save=True)

    print("\n--- 📊 テスト結果 ---")
    for r in results:
        filename = os.path.basename(r.path)
        detected_labels = []
        for box in r.boxes:
            # クラス名を取得（今回は 0:bus, 1:car, 2:traffic light の3つだけ知っている状態です）
            class_id = int(box.cls[0])
            label_name = model.names[class_id]
            detected_labels.append(label_name)
            
        print(f" 📸 {filename}: {detected_labels}")

    print("---------------------------------------------------------")
    print("✅ 完了！ 'runs/detect/predict...' フォルダに枠付きの画像が保存されました。")

if __name__ == "__main__":
    main()