import os
import json
import shutil

# --- 設定 ---
# 1. BDD100Kをダウンロード・解凍したフォルダのパス（ご自身の環境に合わせて変更してください）
BDD_IMG_DIR = "bdd100k/images/100k/train"
BDD_LABEL_JSON = "bdd100k/labels/bdd100k_labels_images_train.json"

# 2. 抽出した画像の保存先（先ほどのブレンド処理の「素材フォルダ」として使います）
SAVE_DIR = "img_bdd_bus"

# 3. 抽出する最大枚数
MAX_IMAGES = 100
# -----------

def main():
    if not os.path.exists(BDD_IMG_DIR) or not os.path.exists(BDD_LABEL_JSON):
        print("エラー: BDD100Kの画像フォルダ、またはラベルJSONが見つかりません。")
        print("設定のパスが正しいか確認してください。")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    print("ラベルデータを読み込んでいます（少し時間がかかります）...")
    
    with open(BDD_LABEL_JSON, 'r') as f:
        bdd_data = json.load(f)

    print("条件（バスが写っている ＋ 雨 または 夜）に合う画像を検索中...")
    extracted_count = 0

    for item in bdd_data:
        # 終了条件
        if extracted_count >= MAX_IMAGES:
            break

        # 1. 画像の属性（天候や時間帯）を取得
        attributes = item.get('attributes', {})
        weather = attributes.get('weather', '')
        timeofday = attributes.get('timeofday', '')

        # 条件: 「雨(rainy)」 または 「夜(night)」 でない場合はスキップ
        if weather != 'rainy' and timeofday != 'night':
            continue

        # 2. 画像に写っている物体（ラベル）を確認
        has_bus = False
        labels = item.get('labels', [])
        for label in labels:
            if label.get('category') == 'bus':
                has_bus = True
                break

        # 3. 条件を満たしていればコピー
        if has_bus:
            img_filename = item['name']
            src_path = os.path.join(BDD_IMG_DIR, img_filename)
            dest_path = os.path.join(SAVE_DIR, f"bdd_{weather}_{timeofday}_{img_filename}")

            if os.path.exists(src_path) and not os.path.exists(dest_path):
                shutil.copy2(src_path, dest_path)
                extracted_count += 1
                
                if extracted_count % 10 == 0:
                    print(f"  ... {extracted_count} 枚抽出完了 (最新: {weather}, {timeofday})")

    print(f"\n抽出完了！ {extracted_count} 枚の過酷な環境のバス画像が '{SAVE_DIR}' に保存されました。")

if __name__ == "__main__":
    main()