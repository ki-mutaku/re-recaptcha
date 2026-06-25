import os
import random
import shutil
from collections import defaultdict

# --- 設定 ---
# データセットの大元フォルダ
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# 分割するクラスのリスト
CLASSES = ["bus", "other"]

# 検証用(val)に回す割合（20% = 0.2）
VAL_RATIO = 0.2

# 同じ元画像から作った加工バリエーションを表すサフィックス。
# original/night/rain は同じ景色なので、train と val に分かれるとデータリークになる。
VARIANT_SUFFIXES = ("_original.jpg", "_night.jpg", "_rain.jpg")
# -----------


def base_key(filename):
    """
    加工バリエーションをまとめるためのキー（元画像ID部分）を返す。

    例: "bus_000000012345_rain.jpg" -> "bus_000000012345"
    サフィックスが無いファイルは、その名前自体をキーにする。
    """
    for suffix in VARIANT_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


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

        # 元画像ID単位でバリエーションをまとめる（リーク防止の肝）
        groups = defaultdict(list)
        for img in images:
            groups[base_key(img)].append(img)

        group_keys = list(groups.keys())

        # 元画像ID単位でシャッフルしてから val に回すグループを選ぶ。
        # これで original/night/rain が必ず train か val のどちらか一方にまとまる。
        random.seed(42)  # いつ実行してもランダム性が一定になるようにシードを固定
        random.shuffle(group_keys)

        val_group_count = int(len(group_keys) * VAL_RATIO)
        val_keys = group_keys[:val_group_count]

        # 実際の移動処理 (copyではなく、trainからvalへmoveします)
        print(f"【{cls}クラス】の処理中...")
        val_moved = 0
        for key in val_keys:
            for img in groups[key]:
                shutil.move(
                    os.path.join(train_cls_dir, img),
                    os.path.join(val_cls_dir, img),
                )
                val_moved += 1

        # 結果の報告
        train_remain = total_images - val_moved
        print(
            f"  -> 元画像 {len(group_keys)} 枚のうち {val_group_count} 枚分（{val_moved} ファイル）を val へ移動しました。"
        )
        print(f"  -> train に {train_remain} ファイル残りました。\n")

    print("すべての分割作業が完了しました！")
    print(f"理想的な '{BASE_DIR}' のディレクトリ構成が完成しました。")

if __name__ == "__main__":
    main()