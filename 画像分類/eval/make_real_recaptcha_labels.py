"""
real_recaptcha/{bus,nonbus}/ のフォルダ構成から正解ラベルCSVを自動生成する。

download_real_recaptcha.py がフォルダ分けの時点でラベル付け済みなので、
人手でラベリングする img_labels.csv 等とは違い、フォルダ名から機械的に作れる。

出力: eval/labels/real_recaptcha_labels.csv（image_path,has_bus）
  image_path は "bus/xxx.png" / "nonbus/xxx.png" の2階層相対パス
  （evaluate.py の match_labels_to_paths が末尾2階層をキーに照合するため）

使い方:
  python 画像分類/eval/make_real_recaptcha_labels.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DIR = os.path.join(HERE, "..", "real_recaptcha")
OUT_CSV = os.path.join(HERE, "labels", "real_recaptcha_labels.csv")


def main():
    bus_dir = os.path.join(REAL_DIR, "bus")
    nonbus_dir = os.path.join(REAL_DIR, "nonbus")
    if not os.path.isdir(bus_dir) or not os.path.isdir(nonbus_dir):
        raise SystemExit(
            f"本物画像が無い: {REAL_DIR}\n"
            "先に eval/download_real_recaptcha.py を実行してください。"
        )

    rows = []
    for fname in sorted(os.listdir(bus_dir)):
        rows.append((f"bus/{fname}", True))
    for fname in sorted(os.listdir(nonbus_dir)):
        rows.append((f"nonbus/{fname}", False))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "has_bus"])
        writer.writerows(rows)

    n_pos = sum(1 for _, v in rows if v)
    n_neg = sum(1 for _, v in rows if not v)
    print(f"[done] {OUT_CSV}")
    print(f"  正例(bus) {n_pos} 枚 / 負例(nonbus) {n_neg} 枚")


if __name__ == "__main__":
    main()
