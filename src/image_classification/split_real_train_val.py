"""
real_recaptcha/ タイルを train / val / test に分割する。

シンボリックリンクでデータをコピーせずに分割するため、
ディスク消費を最小限に抑える。

分割比率:
  train 70% → 学習（勾配更新）に使う
  val   15% → 学習中のモデル選択（ベストF1の記録）に使う
  test  15% → 最終評価専用。学習には一切使わない。

実行方法（リポジトリのルートで）:
  uv run python 画像分類/split_real_train_val.py
"""

import os
import random
from pathlib import Path

# --- 設定 ---
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 残り 15%

REAL_BUS_DIR   = Path("画像分類/real_recaptcha/bus")
REAL_NONBUS_DIR = Path("画像分類/real_recaptcha/nonbus")
OUT_BASE = Path("dataset_real")

# real_recaptcha の "nonbus" → ImageFolder 用に "other" へ
CLASSES = [
    (REAL_BUS_DIR,    "bus"),
    (REAL_NONBUS_DIR, "other"),
]
# -----------


def main():
    random.seed(SEED)

    for src_dir, cls_name in CLASSES:
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
        random.shuffle(files)

        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": files[:n_train],
            "val":   files[n_train : n_train + n_val],
            "test":  files[n_train + n_val :],
        }

        for split_name, split_files in splits.items():
            out_dir = OUT_BASE / split_name / cls_name
            out_dir.mkdir(parents=True, exist_ok=True)

            created = 0
            for f in split_files:
                src = src_dir.resolve() / f
                dst = out_dir / f
                if not dst.exists():
                    os.symlink(src, dst)
                    created += 1

            print(f"  {cls_name:5s} / {split_name:5s}: {len(split_files):5d} タイル"
                  f"  (新規リンク: {created})")

    print("\n完了！")
    print("  学習: uv run python 画像分類/train_resnet.py"
          " --data-dir dataset_real --save-prefix best_resnet18_bus_real")


if __name__ == "__main__":
    main()
