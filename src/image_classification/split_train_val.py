import os
import random
import shutil

# --- 設定 ---
# データセットの大元フォルダ
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# 分割するクラスのリスト
CLASSES = ["bus", "other"]

# 検証用(val)に回す割合（20% = 0.2）
VAL_RATIO = 0.2
# -----------

def main():
    print("学習用(train)と検証用(val)のデータ分割を開始します...\n")

    for cls in CLASSES:
        train_cls_dir = os.path.join(TRAIN_DIR, cls)
        val_cls_dir = os.path.join(VAL_DIR, cls)

        # val用のフォルダが存在しない場合は作成
        os.makedirs(val_cls_dir, exist_ok=True)

        if not os.path.exists(train_cls_dir):
            print(f"⚠️ 警告: フォルダが見つかりません -> {train_cls_dir}")
            continue

        # フォルダ内の画像リストを取得
        images = [f for f in os.listdir(train_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_images = len(images)

        if total_images == 0:
            print(f"⚠️ 警告: '{cls}' フォルダに画像がありません。")
            continue

        # 移動させる枚数を計算 (全画像の20%)
        val_count = int(total_images * VAL_RATIO)

        # 画像リストをランダムにシャッフル（原本・夜・雨を均等にばらけさせるため）
        random.seed(42) # いつ実行してもランダム性が一定になるようにシードを固定
        random.shuffle(images)

        # valに移動させる画像を抽出
        val_images = images[:val_count]

        # 実際の移動処理 (copyではなく、trainからvalへmoveします)
        print(f"【{cls}クラス】の処理中...")
        for img in val_images:
            src_path = os.path.join(train_cls_dir, img)
            dst_path = os.path.join(val_cls_dir, img)
            shutil.move(src_path, dst_path)

        # 結果の報告
        train_remain = total_images - val_count
        print(f"  -> 全 {total_images} 枚のうち、{val_count} 枚を val へ移動しました。")
        print(f"  -> train に {train_remain} 枚残りました。\n")

    print("すべての分割作業が完了しました！")
    print(f"理想的な '{BASE_DIR}' のディレクトリ構成が完成しました。")

if __name__ == "__main__":
    main()