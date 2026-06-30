"""
モデル比較スクリプト (compare_models.py)

- 役割: zero-shot ResNet（ImageNet事前学習のまま）と
        ファインチューニング後ResNet（best_resnet18_bus.pth）を、
        本物の評価データで「同じPR-AUC(AP)軸」で並べて比較する。
- 評価データ:
    - img         … 晴れ（劣化なし）
    - img_bus_rain … 雨（劣化あり）
  の2セットに対して、それぞれ両モデルのAPを出す。
  これで「FTに意味があったか」「雨でどちらが強いか」を判定できる。

- 出力: コンソールの比較表 + マークダウンレポート + PR曲線重ね描きPNG（--output-dir）

実行（リポジトリのルートから）:
    uv run python 画像分類/eval/compare_models.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 同ディレクトリの evaluate.py（評価ロジック一式）と、1つ上の classification.py を使う
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from classification import (  # noqa: E402
    build_preprocess,
    collect_image_paths,
    load_resnet_classifier,
    predict_class_probabilities,
    target_confidence_score,
)
from evaluate import (  # noqa: E402
    compute_average_precision,
    load_labels,
    match_labels_to_paths,
    sweep_thresholds,
)

# パスはリポジトリのルート基準で解決する（どこから実行しても動くように）
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = Path(__file__).resolve().parent / "labels"
FT_MODEL_PATH = REPO_ROOT / "best_resnet18_bus.pth"
FT_CLASSES_PATH = REPO_ROOT / "best_resnet18_bus_classes.json"

# 評価する3セット: (画像ディレクトリ, ラベルCSV, 表示名)
DATASETS = [
    (REPO_ROOT / "img", LABELS_DIR / "img_labels.csv", "晴れ (img)"),
    (REPO_ROOT / "img_bus_rain", LABELS_DIR / "img_bus_rain_labels.csv", "雨 (img_bus_rain)"),
    (
        REPO_ROOT / "画像分類" / "real_recaptcha",
        LABELS_DIR / "real_recaptcha_labels.csv",
        "本物 (real_recaptcha)",
    ),
]

DEFAULT_TARGET = "bus"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def zeroshot_scores(image_paths, target):
    """
    zero-shot ResNet で各画像の「busである確信度スコア」を求める。

    ImageNet 1000クラスのsoftmax確率のうち、busに属するクラスの確率を合計する
    （classification.py の target_confidence_score と同じ定義）。前処理はパディング。
    """
    model, categories = load_resnet_classifier()
    preprocess = build_preprocess()
    scores = {}
    for idx, path in enumerate(image_paths):
        probabilities = predict_class_probabilities(path, model, preprocess)
        scores[idx] = target_confidence_score(probabilities, categories, target)
    return scores


def build_ft_preprocess():
    """
    FTモデル用の前処理。train_resnet.py の val と同じ（Resizeで224四方＋正規化）。

    学習時と同じ前処理で評価しないと不公平になるため、こちらは中央寄せのResizeを使う
    （zero-shot側のパディングとは別物だが、各モデルを学習時の条件で測るのが公平）。
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_ft_model(model_path, classes_path):
    """
    ファインチューニング済みResNet18（2クラス: bus/other）を読み込む。

    最終層を学習時と同じ形（クラス数ぶんの全結合層）に付け替えてから重みを流し込む。
    クラス名の並びは学習時に保存した JSON から復元する。
    """
    with open(classes_path, encoding="utf-8") as f:
        class_names = json.load(f)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, class_names


def ft_scores(image_paths, model, class_names, preprocess, target):
    """
    FTモデルで各画像の「busである確信度スコア」= softmax確率の bus 成分を求める。

    二値分類なので、busクラスの確率がそのまま確信度になる。
    zero-shot と同じ 0〜1 のスコアなので、同じAP軸で比較できる。
    """
    bus_idx = class_names.index(target)
    scores = {}
    for idx, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(input_tensor)[0], dim=0)
        scores[idx] = probabilities[bus_idx].item()
    return scores


def key_metrics(scores, labeled_data):
    """
    APだけでは見えない実用的な数値をまとめて返す。

    - best_f1 / best_f1_threshold … F1が最大になる点とそのときの閾値
    - recall_at_p1 … 適合率を1.0(誤検出ゼロ)に保ったまま到達できる最大の再現率
    - precision_at_r1 … 再現率1.0(バスを全部拾う)を満たすときの最大の適合率
    """
    distinct = sorted({scores[idx] for idx, _ in labeled_data}, reverse=True)
    rows = sweep_thresholds(scores, labeled_data, distinct)

    best_f1, best_t = 0.0, None
    for r in rows:
        p, rec = r["precision"], r["recall"]
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, r["threshold"]

    recall_at_p1 = max((r["recall"] for r in rows if r["precision"] >= 0.999), default=0.0)
    precision_at_r1 = max((r["precision"] for r in rows if r["recall"] >= 0.999), default=0.0)

    return {
        "best_f1": best_f1,
        "best_f1_threshold": best_t,
        "recall_at_p1": recall_at_p1,
        "precision_at_r1": precision_at_r1,
    }


def pr_curve_points(scores, labeled_data):
    """
    観測された全スコアを閾値にして、滑らかなPR曲線の (recall, precision) 点列を返す。
    """
    distinct = sorted({scores[idx] for idx, _ in labeled_data}, reverse=True)
    rows = sweep_thresholds(scores, labeled_data, distinct)
    recall = [r["recall"] for r in rows]
    precision = [r["precision"] for r in rows]
    return recall, precision


def evaluate_one(image_dir, labels_path, target):
    """
    1つのデータセットに対して、zero-shot と FT 両方のAP・PR曲線を計算して返す。
    """
    image_paths = collect_image_paths(image_dir)
    label_dict = load_labels(labels_path)
    labeled_data = match_labels_to_paths(image_paths, label_dict)

    total_pos = sum(1 for _, is_pos in labeled_data if is_pos)
    total_neg = sum(1 for _, is_pos in labeled_data if not is_pos)

    # zero-shot
    zs = zeroshot_scores(image_paths, target)
    zs_ap = compute_average_precision(zs, labeled_data)
    zs_curve = pr_curve_points(zs, labeled_data)
    zs_key = key_metrics(zs, labeled_data)

    # FT
    ft_model, class_names = load_ft_model(FT_MODEL_PATH, FT_CLASSES_PATH)
    ft_pre = build_ft_preprocess()
    ft = ft_scores(image_paths, ft_model, class_names, ft_pre, target)
    ft_ap = compute_average_precision(ft, labeled_data)
    ft_curve = pr_curve_points(ft, labeled_data)
    ft_key = key_metrics(ft, labeled_data)

    return {
        "n_pos": total_pos,
        "n_neg": total_neg,
        "zs_ap": zs_ap,
        "ft_ap": ft_ap,
        "zs_curve": zs_curve,
        "ft_curve": ft_curve,
        "zs_key": zs_key,
        "ft_key": ft_key,
    }


def plot_overlay(result, title, output_path):
    """
    zero-shot と FT のPR曲線を1枚に重ねて保存する（面積が大きいほど高性能）。
    """
    plt.figure(figsize=(6, 5))
    zr, zp = result["zs_curve"]
    fr, fp = result["ft_curve"]
    plt.plot(zr, zp, "b-", label=f"zero-shot (AP={result['zs_ap']:.3f})")
    plt.plot(fr, fp, "r-", label=f"fine-tuned (AP={result['ft_ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="zero-shot ResNet と FT後ResNet を本物データで同じAP軸で比較します。"
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="探したい対象")
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="出力ディレクトリ"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not FT_MODEL_PATH.exists():
        print(f"エラー: FTモデルが見つかりません -> {FT_MODEL_PATH}")
        print("先に train_resnet.py を実行してください。")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    report_lines = [
        "# zero-shot ResNet vs ファインチューニング後 比較",
        "",
        "本物の評価データ（晴れ=img / 雨=img_bus_rain）に対して、",
        "ImageNet事前学習のままのResNet（zero-shot）と、",
        "bus/other 2クラスに微調整したResNet（fine-tuned）を、",
        "**同じ PR-AUC (Average Precision) 軸**で比較した結果。",
        "",
        "| データセット | 正例/負例 | zero-shot AP | fine-tuned AP | 差分(FT−ZS) |",
        "|---|---:|---:|---:|---:|",
    ]

    for image_dir, labels_path, display in DATASETS:
        if not Path(image_dir).exists():
            print(f"スキップ: 画像ディレクトリが無い -> {image_dir}")
            continue
        print(f"\n=== {display} を評価中 ===")
        res = evaluate_one(image_dir, labels_path, args.target)

        diff = res["ft_ap"] - res["zs_ap"]
        zk, fk = res["zs_key"], res["ft_key"]
        print(
            f"  正例 {res['n_pos']} / 負例 {res['n_neg']}  "
            f"zero-shot AP={res['zs_ap']:.3f}  fine-tuned AP={res['ft_ap']:.3f}  (差 {diff:+.3f})"
        )
        print(
            f"    zero-shot : maxF1={zk['best_f1']:.3f}(閾値{zk['best_f1_threshold']:.3f})  "
            f"適合率1.0での再現率={zk['recall_at_p1']:.3f}  再現率1.0での適合率={zk['precision_at_r1']:.3f}"
        )
        print(
            f"    fine-tuned: maxF1={fk['best_f1']:.3f}(閾値{fk['best_f1_threshold']:.3f})  "
            f"適合率1.0での再現率={fk['recall_at_p1']:.3f}  再現率1.0での適合率={fk['precision_at_r1']:.3f}"
        )

        chart = output_dir / f"{Path(image_dir).name}_zs_vs_ft_PR.png"
        # グラフのタイトルは日本語フォント不足で化けるため、ディレクトリ名(ASCII)を使う
        plot_overlay(res, f"{Path(image_dir).name}: zero-shot vs fine-tuned", chart)
        print(f"  PR曲線を保存: {chart}")

        report_lines.append(
            f"| {display} | {res['n_pos']}/{res['n_neg']} "
            f"| {res['zs_ap']:.3f} | {res['ft_ap']:.3f} | {diff:+.3f} |"
        )
        table_rows.append((display, res, chart))

    report_lines += [
        "",
        "## 詳しい指標",
        "",
        "AP だけでは見えない実用面の数値も並べる。",
        "「適合率1.0での再現率」は、誤検出ゼロのまま何割のバスを拾えるか。",
        "「再現率1.0での適合率」は、バスを全部拾おうとしたとき選択がどれだけ綺麗か。",
        "",
        "| データセット | モデル | AP | 最大F1 (閾値) | 適合率1.0での再現率 | 再現率1.0での適合率 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for display, res, _chart in table_rows:
        for name, key in (("zero-shot", "zs_key"), ("fine-tuned", "ft_key")):
            k = res[key]
            ap = res["zs_ap"] if key == "zs_key" else res["ft_ap"]
            report_lines.append(
                f"| {display} | {name} | {ap:.3f} "
                f"| {k['best_f1']:.3f} ({k['best_f1_threshold']:.3f}) "
                f"| {k['recall_at_p1']:.3f} | {k['precision_at_r1']:.3f} |"
            )

    report_lines += [
        "",
        "## PR曲線",
        "",
    ]
    for display, _res, chart in table_rows:
        report_lines += [f"### {display}", "", f"![PR]({Path(chart).name})", ""]

    report_lines += [
        "## 読み方",
        "",
        "- AP（PR-AUCの値）が大きいほど高性能。1.0が理想。",
        "- 差分が正なら fine-tuning で改善、負なら zero-shot の方が強い。",
        "- 晴れと雨で差を比べると、劣化画像に対する強さの違いが分かる。",
        "- zero-shot と fine-tuned は前処理が異なる（zero-shot=パディング / FT=Resize）。",
        "  各モデルを学習時と同じ前処理で測ることで、公平な比較にしている。",
    ]

    report_path = output_dir / "zeroshot_vs_ft_比較.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print("\n===== 比較サマリ =====")
    for display, res, _chart in table_rows:
        print(
            f"  {display:<22} zero-shot AP={res['zs_ap']:.3f}  "
            f"fine-tuned AP={res['ft_ap']:.3f}  (差 {res['ft_ap'] - res['zs_ap']:+.3f})"
        )
    print(f"\nレポートを保存しました: {report_path}")


if __name__ == "__main__":
    main()
