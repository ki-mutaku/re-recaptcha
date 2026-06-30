"""
評価スクリプト (evaluate.py)

- モデル: src/classification.py のResNet18を使用
- 役割: 正解ラベルと照合して、確信度スコアの閾値スイープで適合率・再現率を計算し、
        PR曲線・PR-AUC(AP)・報告書を出力する
- 入力: 画像ディレクトリ + 正解ラベルCSV (image_path, has_bus)
- 出力: 評価指標CSV・グラフPNG・マークダウンレポート (すべて --output-dir に保存)

評価方法は「バス確信度スコアの閾値スイープ」に統一している。
スコアが閾値以上の画像を選択とみなしてPR曲線を描き、その下側の面積を
PR-AUC (Average Precision) とする。閾値に依存しない1つの指標なので、
生ResNetとファインチューニング後を同じ土俵(同じAP)で比較できる。

実行例:
    uv run python eval/evaluate.py --image-dir img --labels eval/labels/img_labels.csv
    uv run python eval/evaluate.py --image-dir img_bus_rain --labels eval/labels/img_bus_rain_labels.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# classification.py は eval/ の1つ上（画像分類/ 直下）に置かれている
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification import (
    build_preprocess,
    collect_image_paths,
    load_resnet_classifier,
    predict_class_probabilities,
    target_confidence_score,
)

DEFAULT_LABELS = "eval/labels/img_labels.csv"
DEFAULT_TARGET = "bus"
DEFAULT_OUTPUT_DIR = "eval/results"

# 閾値スイープのテーブル・折れ線グラフに使う、読みやすい閾値の並び。
# バス確信度は正例で大きく負例で0付近になるため、低めの値も細かく刻む。
THRESHOLD_GRID = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0]


def load_labels(labels_path):
    """
    正解ラベルCSVを読み込み、{image_path文字列: bool} の辞書を返す。

    CSVの形式: image_path,has_bus
    image_path は "img/1.jpg" のような相対パスを想定している。
    """
    label_dict = {}
    with open(labels_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["image_path"].strip().replace("\\", "/")
            label_dict[key] = row["has_bus"].strip().lower() == "true"
    return label_dict


def match_labels_to_paths(image_paths, label_dict):
    """
    image_paths の各パスを正解ラベルと照合し、ラベルが付いている画像だけを返す。

    ラベルのキーは "img/1.jpg" のような2階層の相対パスを想定している。
    絶対パスや異なる区切り文字でも最後の2階層を取り出して照合する。
    """
    labeled = []
    for idx, path in enumerate(image_paths):
        normalized = str(path).replace("\\", "/")
        # "img/1.jpg" のように2階層のキーを作って照合する
        parts = normalized.split("/")
        short_key = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        if short_key in label_dict:
            labeled.append((idx, label_dict[short_key]))
    return labeled


def compute_metrics(selected_indices, labeled_data):
    """
    TP, FP, FN, TN, 適合率(precision), 再現率(recall) を計算する。

    selected_indices: モデルが選択したインデックスのリスト
    labeled_data: [(index, is_positive), ...] の形式
    """
    selected_set = set(selected_indices)
    tp = sum(1 for idx, is_pos in labeled_data if idx in selected_set and is_pos)
    fp = sum(1 for idx, is_pos in labeled_data if idx in selected_set and not is_pos)
    fn = sum(1 for idx, is_pos in labeled_data if idx not in selected_set and is_pos)
    tn = sum(1 for idx, is_pos in labeled_data if idx not in selected_set and not is_pos)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}


def get_all_scores(image_paths, target):
    """
    全画像について「対象が写っている確信度スコア(0〜1)」を計算して {index: score} を返す。

    モデルは1回だけ読み込む。各画像のsoftmax確率のうち、対象の大分類に
    属するクラスの確率を合計してスコアにする (target_confidence_score)。
    """
    model, categories = load_resnet_classifier()
    preprocess = build_preprocess()
    scores = {}
    for idx, path in enumerate(image_paths):
        probabilities = predict_class_probabilities(path, model, preprocess)
        scores[idx] = target_confidence_score(probabilities, categories, target)
        print(f"  [{idx + 1}/{len(image_paths)}] {path.name}: score={scores[idx]:.3f}")
    return scores


def sweep_thresholds(scores, labeled_data, thresholds):
    """
    閾値を変えながら各指標を収集する。

    スコアが閾値以上の画像を「選択」とみなして判定する。
    閾値を下げるほど選択が増え、再現率が上がる一方で適合率は下がりやすい。
    """
    results = []
    for t in thresholds:
        selected = [idx for idx, score in scores.items() if score >= t]
        metrics = compute_metrics(selected, labeled_data)
        metrics["threshold"] = t
        metrics["n_selected"] = len(selected)
        results.append(metrics)
    return results


def compute_average_precision(scores, labeled_data):
    """
    PR-AUC (Average Precision) を計算する。

    ラベル付き画像をスコアの高い順に並べ、1枚ずつ選択を増やしながら
    precision × (recallの増分) を足し合わせる。PR曲線の下側の面積に相当し、
    閾値に依存しない1つの数値としてモデル全体の性能を表す (1.0が理想)。
    """
    ranked = sorted(
        ((scores[idx], is_pos) for idx, is_pos in labeled_data),
        key=lambda pair: pair[0],
        reverse=True,
    )
    total_pos = sum(1 for _, is_pos in labeled_data if is_pos)
    if total_pos == 0:
        return 0.0

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _, is_pos in ranked:
        if is_pos:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / total_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def save_metrics_csv(results, output_path):
    """
    閾値スイープの評価指標をCSVに保存する。F1スコアも計算して加える。
    """
    fields = [
        "threshold", "tp", "fp", "fn", "tn",
        "n_selected", "precision", "recall", "f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            p, r = row["precision"], row["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            out = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()}
            out["f1"] = f"{f1:.4f}"
            writer.writerow(out)


def plot_results(
    grid_results, curve_results, ap, output_dir, title_prefix="", filename_prefix=""
):
    """
    評価グラフ2枚を1つのPNGとして保存する。

    左: 閾値を動かしたときの適合率・再現率の推移 (THRESHOLD_GRID の各点)。
    右: 全スコアを閾値にした滑らかなPR曲線。タイトルにPR-AUC(AP)を表示する。
    PR曲線はFT後のモデルでも同じ軸で描けるため、生 vs FT の比較に使える。
    """
    thresholds = [r["threshold"] for r in grid_results]
    grid_precision = [r["precision"] for r in grid_results]
    grid_recall = [r["recall"] for r in grid_results]
    curve_recall = [r["recall"] for r in curve_results]
    curve_precision = [r["precision"] for r in curve_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 折れ線グラフ: threshold vs precision / recall
    axes[0].plot(thresholds, grid_precision, "b-o", label="Precision")
    axes[0].plot(thresholds, grid_recall, "r-s", label="Recall")
    axes[0].set_xlabel("threshold (bus confidence)")
    axes[0].set_ylabel("Score")
    axes[0].set_title(f"{title_prefix}Precision / Recall vs threshold")
    axes[0].set_ylim(0, 1.05)
    axes[0].invert_xaxis()  # 閾値が高い(厳しい)→低い(緩い)の順に左から並べる
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR曲線: recall を x 軸、precision を y 軸。下側の面積がAP。
    axes[1].plot(curve_recall, curve_precision, "g-")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix}PR Curve (AP = {ap:.3f})")
    axes[1].set_xlim(-0.05, 1.05)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = Path(output_dir) / f"{filename_prefix}precision_recall_chart.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"グラフを保存しました: {chart_path}")
    return chart_path


def generate_report(grid_results, ap, image_dir, target, labels_path, output_path, chart_path):
    """
    閾値スイープの結果をマークダウンレポートにまとめて保存する。
    """
    total_pos = max(r["tp"] + r["fn"] for r in grid_results)
    total_neg = max(r["fp"] + r["tn"] for r in grid_results)
    total = total_pos + total_neg

    best_f1, best_f1_t = 0.0, grid_results[0]["threshold"]
    for r in grid_results:
        p, rec = r["precision"], r["recall"]
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        r["f1"] = f1
        if f1 > best_f1:
            best_f1, best_f1_t = f1, r["threshold"]

    chart_rel = Path(chart_path).name

    lines = [
        "# 分類モデル評価結果 (閾値スイープ)",
        "",
        "## 実験条件",
        "",
        f"- 画像ディレクトリ: `{image_dir}`",
        f"- 正解ラベル: `{labels_path}`",
        f"- お題: `{target}`",
        "- モデル: ResNet18 (ImageNet事前学習済み)",
        f"- 評価画像数: {total} 枚（正例 {total_pos} 枚 / 負例 {total_neg} 枚）",
        f"- **PR-AUC (Average Precision): {ap:.3f}**",
        "",
        "## 評価指標の定義",
        "",
        "| 指標 | 定義 | 計算式 |",
        "|---|---|---|",
        "| TP (真陽性) | バスが写っている画像をモデルが正しく選択した枚数 | — |",
        "| FP (偽陽性) | バスが写っていない画像をモデルが誤って選択した枚数 | — |",
        "| FN (偽陰性) | バスが写っている画像をモデルが見逃した枚数 | — |",
        "| TN (真陰性) | バスが写っていない画像をモデルが正しく除外した枚数 | — |",
        "| 適合率 (Precision) | 選択した中に実際にバスが写っていた割合 | TP / (TP + FP) |",
        "| 再現率 (Recall) | バス画像全体のうちモデルが漏らさず選択できた割合 | TP / (TP + FN) |",
        "| PR-AUC (AP) | 閾値を動かして描いたPR曲線の下側の面積（閾値非依存の総合性能） | — |",
        "",
        "## 評価方法",
        "",
        "各画像について「バスである確信度スコア (0〜1)」を計算し、",
        "スコアが閾値以上の画像を選択とみなして適合率・再現率を求めた。",
        "閾値を動かして描いたPR曲線の下側の面積が PR-AUC (AP) であり、",
        "閾値に依存しない総合性能としてモデル同士の比較に使える。",
        "この評価軸はファインチューニング後のモデルでも同じく使えるため、",
        "生ResNetとFT後を同じPR曲線・同じAPで比較できる。",
        "",
        "## 閾値別の評価結果",
        "",
        "| 閾値 | TP | FP | FN | TN | 選択数 | 適合率 | 再現率 | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for r in grid_results:
        f1 = r.get("f1", 0.0)
        lines.append(
            f"| {r['threshold']:.2f} "
            f"| {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} "
            f"| {r['n_selected']} "
            f"| {r['precision']:.3f} | {r['recall']:.3f} | {f1:.3f} |"
        )

    lines += [
        "",
        "## グラフ",
        "",
        f"![PR曲線]({chart_rel})",
        "",
        "左: 閾値を変えたときの適合率・再現率の推移。  ",
        "右: 適合率-再現率曲線（左上に近いほど高性能、面積=AP）。",
        "",
        "## 考察",
        "",
        f"- PR-AUC (AP) = **{ap:.3f}**（閾値に依存しないモデル全体の性能）。",
        f"- F1スコアが最大になる閾値: **{best_f1_t:.2f}** (F1 = {best_f1:.3f})。",
        "- 閾値を下げると再現率は上がるが適合率は下がる（精度とカバレッジのトレードオフ）。",
        "- FT後のモデルも同じ閾値スイープで評価すれば、APの大小で改善幅を比較できる。",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"レポートを保存しました: {output_path}")


def parse_args():
    """
    コマンドライン引数を読み取る。
    """
    parser = argparse.ArgumentParser(
        description="分類モデルの適合率・再現率を閾値スイープで評価し、PR曲線とレポートを生成します。"
    )
    parser.add_argument("--image-dir", default="img", help="評価する画像ディレクトリ")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="正解ラベルCSVのパス")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="探したい対象")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="出力ディレクトリ")
    return parser.parse_args()


def main():
    """
    スクリプトとして実行されたときの入口。
    """
    args = parse_args()

    image_paths = collect_image_paths(args.image_dir)
    if not image_paths:
        print(f"画像が見つかりませんでした: {args.image_dir}")
        return

    label_dict = load_labels(args.labels)
    labeled_data = match_labels_to_paths(image_paths, label_dict)

    if not labeled_data:
        print(f"ラベルと一致する画像が見つかりませんでした。labels: {args.labels}")
        return

    total_pos = sum(1 for _, is_pos in labeled_data if is_pos)
    total_neg = sum(1 for _, is_pos in labeled_data if not is_pos)
    print(
        f"評価対象: {len(labeled_data)} 枚 (正例 {total_pos} / 負例 {total_neg}) "
        f"[全 {len(image_paths)} 枚中]"
    )

    print("\n--- スコア計算中 ---")
    scores = get_all_scores(image_paths, args.target)

    grid_results = sweep_thresholds(scores, labeled_data, THRESHOLD_GRID)
    # 滑らかなPR曲線とAPは、観測された全スコアを閾値にして求める
    distinct_thresholds = sorted({scores[idx] for idx, _ in labeled_data}, reverse=True)
    curve_results = sweep_thresholds(scores, labeled_data, distinct_thresholds)
    ap = compute_average_precision(scores, labeled_data)

    print(f"\n--- 閾値スイープ (PR-AUC / AP = {ap:.3f}) ---")
    for r in grid_results:
        print(
            f"  threshold={r['threshold']:.2f}: precision={r['precision']:.3f}  "
            f"recall={r['recall']:.3f}  n_selected={r['n_selected']}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.image_dir).name + "_"

    metrics_csv = output_dir / f"{prefix}evaluation_metrics.csv"
    save_metrics_csv(grid_results, metrics_csv)
    print(f"指標CSVを保存しました: {metrics_csv}")

    chart_path = plot_results(
        grid_results, curve_results, ap, output_dir,
        title_prefix=f"[{args.image_dir}] ",
        filename_prefix=prefix,
    )
    report_path = output_dir / f"{prefix}評価結果まとめ.md"
    generate_report(
        grid_results, ap, args.image_dir, args.target, args.labels, report_path, chart_path
    )


if __name__ == "__main__":
    main()
