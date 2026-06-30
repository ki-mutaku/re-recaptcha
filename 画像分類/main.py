"""
フィルタ妥当性検証の一連を、ゼロから一括実行するエントリポイント。

実行内容（順番）:
  1. 本物 reCAPTCHA 画像のダウンロード（無ければ取得。bus/nonbus）
  2. フィルタ妥当性の定量評価（FID / ドメイン分類器AUC）
  3. 結果 CSV の場所を表示

前提:
  - 依存導入済み（uv sync もしくは
    pip install datasets scipy scikit-learn torch torchvision）
  - クリーン bus 画像 busbus/ はリポジトリに同梱済み

使い方（リポジトリのどこからでも可）:
  python 画像分類/main.py              # 本物は全件DL
  python 画像分類/main.py --n 600      # 本物を各600枚に制限（軽い）
  python 画像分類/main.py --force-download   # 既存があっても再取得
"""

import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "eval")
REAL_BUS_DIR = os.path.join(HERE, "real_recaptcha", "bus")
PY = sys.executable


def run(script, extra_args):
    cmd = [PY, os.path.join(EVAL_DIR, script), *extra_args]
    print(f"\n$ {' '.join(cmd)}\n" + "-" * 60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=0,
                        help="本物の取得枚数（0=全件）")
    parser.add_argument("--force-download", action="store_true",
                        help="既に画像があっても再ダウンロードする")
    args = parser.parse_args()

    # ステップ1: 本物画像のダウンロード（無ければ）
    already = glob.glob(os.path.join(REAL_BUS_DIR, "*.png"))
    if already and not args.force_download:
        print(f"[skip] 本物画像が既に {len(already)} 枚あるのでDLを省略 "
              f"(--force-download で再取得)")
    else:
        print("[1/2] 本物 reCAPTCHA 画像をダウンロード")
        run("download_real_recaptcha.py", ["--n", str(args.n)])

    # ステップ2: フィルタ妥当性の定量評価
    print("\n[2/2] フィルタ妥当性の定量評価")
    run("filter_validity.py", [])

    out_csv = os.path.join(EVAL_DIR, "results", "filter_validity.csv")
    print(f"\n✅ 完了。結果: {out_csv}")


if __name__ == "__main__":
    main()
