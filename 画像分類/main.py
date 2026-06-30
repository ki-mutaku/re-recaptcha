"""
学習データDL+加工 → train/val分割 → 学習 → 本物画像DL → フィルタ妥当性評価を一括実行。

ステップ（README.md の手順と一致させている）:
  1. バス学習データDL+加工   dataset/train/bus/   に300枚（original/night/rain）
  2. other学習データDL+加工  dataset/train/other/ に300枚
  3. train/val分割          dataset/val/{bus,other}/ へ約20%を移動（元画像ID単位でリーク防止）
  4. 学習                   best_resnet18_bus.pth を生成
  5. 本物reCAPTCHA画像DL     画像分類/real_recaptcha/{bus,nonbus}/ を取得
  6. フィルタ妥当性評価       FID/ドメインAUC を測定 → eval/results/filter_validity.csv

スキップ条件（--force なし）:
  1. dataset/train/bus/   に *_original.jpg があればスキップ
  2. dataset/train/other/ に *_original.jpg があればスキップ
  3. dataset/val/bus/     に画像があればスキップ
  4. best_resnet18_bus.pth があればスキップ
  5. 画像分類/real_recaptcha/bus/ に *.png があればスキップ
  6. 常に実行

使い方（リポジトリルートから）:
  uv run python 画像分類/main.py
  uv run python 画像分類/main.py --n 600        # 本物画像を各600枚に制限
  uv run python 画像分類/main.py --force        # 全ステップを強制再実行
  uv run python 画像分類/main.py --from-step 4  # ステップ4から再開（1〜6）
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
EVAL_DIR = os.path.join(HERE, "eval")
PY = sys.executable

DATASET_BUS_DIR = os.path.join(REPO_ROOT, "dataset", "train", "bus")
DATASET_OTHER_DIR = os.path.join(REPO_ROOT, "dataset", "train", "other")
VAL_BUS_DIR = os.path.join(REPO_ROOT, "dataset", "val", "bus")
MODEL_PATH = os.path.join(REPO_ROOT, "best_resnet18_bus.pth")
REAL_BUS_DIR = os.path.join(HERE, "real_recaptcha", "bus")

TOTAL_STEPS = 6


def run(script_path, extra_args=()):
    """スクリプトを REPO_ROOT を CWD として実行する。"""
    cmd = [PY, script_path, *extra_args]
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 60)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def has_files(directory, pattern):
    return bool(glob.glob(os.path.join(directory, pattern)))


def main():
    parser = argparse.ArgumentParser(
        description="学習データ準備 → train/val分割 → 学習 → 本物DL → フィルタ妥当性評価を一括実行"
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="本物 reCAPTCHA の取得枚数上限（0=全件）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存ファイルを無視して全ステップを再実行",
    )
    parser.add_argument(
        "--from-step", type=int, default=1, choices=range(1, TOTAL_STEPS + 1),
        metavar="N", help=f"N ステップ目から再開（1〜{TOTAL_STEPS}）",
    )
    args = parser.parse_args()

    def skip(step, directory, pattern, label):
        """スキップ条件を満たすなら True を返してメッセージを表示する。"""
        if args.force or step < args.from_step:
            return step < args.from_step
        if has_files(directory, pattern):
            print(f"[skip step{step}] {label} に画像あり（--force で再実行）")
            return True
        return False

    # ステップ1: バス学習データ DL+加工
    if not skip(1, DATASET_BUS_DIR, "*_original.jpg", DATASET_BUS_DIR):
        print(f"\n[1/{TOTAL_STEPS}] バス学習データ DL+加工")
        run(os.path.join(HERE, "download_train_bus.py"))

    # ステップ2: other 学習データ DL+加工
    if not skip(2, DATASET_OTHER_DIR, "*_original.jpg", DATASET_OTHER_DIR):
        print(f"\n[2/{TOTAL_STEPS}] other 学習データ DL+加工")
        run(os.path.join(HERE, "download_train_other.py"))

    # ステップ3: train/val 分割（元画像ID単位でリーク防止）
    if not skip(3, VAL_BUS_DIR, "*.jpg", VAL_BUS_DIR):
        print(f"\n[3/{TOTAL_STEPS}] train/val 分割")
        run(os.path.join(HERE, "split_train_val.py"))

    # ステップ4: 学習
    if args.from_step <= 4:
        if not args.force and os.path.exists(MODEL_PATH):
            print(f"[skip step4] {MODEL_PATH} あり（--force で再学習）")
        else:
            print(f"\n[4/{TOTAL_STEPS}] ResNet18 学習")
            run(os.path.join(HERE, "train_resnet.py"))

    # ステップ5: 本物 reCAPTCHA 画像 DL
    if not skip(5, REAL_BUS_DIR, "*.png", REAL_BUS_DIR):
        print(f"\n[5/{TOTAL_STEPS}] 本物 reCAPTCHA 画像 DL")
        extra = ["--n", str(args.n)] if args.n > 0 else []
        run(os.path.join(EVAL_DIR, "download_real_recaptcha.py"), extra)

    # ステップ6: フィルタ妥当性評価（常に実行）
    if args.from_step <= 6:
        print(f"\n[6/{TOTAL_STEPS}] フィルタ妥当性評価")
        run(os.path.join(EVAL_DIR, "filter_validity.py"))

    out_csv = os.path.join(EVAL_DIR, "results", "filter_validity.csv")
    print(f"\n完了。結果: {out_csv}")


if __name__ == "__main__":
    main()
