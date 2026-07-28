"""合成のみFTと本物混合FTを、固定test splitだけで比較する。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent

DEFAULT_MANIFEST = HERE / "splits" / "real_recaptcha_split.csv"
DEFAULT_REAL_DIR = PKG_DIR / "data" / "real_recaptcha"
DEFAULT_BASELINE_MODEL = PKG_DIR / "models" / "best_resnet18_bus.pth"
DEFAULT_MIXED_MODEL = PKG_DIR / "models" / "best_resnet18_bus_mixed_real.pth"
DEFAULT_OUTPUT_DIR = HERE / "results"


class EvaluationDataset(Dataset):
    """test manifestを順序固定で読み、画像tensorとbus真偽を返す。"""

    def __init__(
        self, rows: list[dict[str, str]], data_dir: Path
    ) -> None:
        from compare_models import build_ft_preprocess

        self.rows = rows
        self.data_dir = data_dir
        self.transform = build_ft_preprocess()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool]:
        row = self.rows[index]
        with Image.open(self.data_dir / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["label"] == "bus"


def resolve_device(requested: str) -> torch.device:
    """auto指定をCUDA、MPS、CPUの優先順で解決する。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_and_audit_manifest(path: Path) -> list[dict[str, str]]:
    """固定manifestを読み、近重複groupのsplit漏洩を検査する。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"固定splitがありません: {path}\n"
            "先に prepare_real_recaptcha_split.py を実行してください。"
        )
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image_path", "label", "split", "sha256", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifestの列が不足しています: {required}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"近重複groupがsplitをまたいでいます: {leaking[:5]}")
    return rows


def select_test_rows(
    rows: list[dict[str, str]], max_per_class: int, seed: int
) -> list[dict[str, str]]:
    """testだけから各クラス同数の決定的な評価集合を返す。"""
    selected: list[dict[str, str]] = []
    for label in ("bus", "nonbus"):
        candidates = [
            row for row in rows if row["split"] == "test" and row["label"] == label
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|test|{row['sha256']}".encode()
            ).hexdigest()
        )
        if max_per_class > 0:
            candidates = candidates[:max_per_class]
        selected.extend(candidates)
    selected.sort(key=lambda row: row["image_path"])
    return selected


def score_model(
    model_path: Path,
    classes_path: Path,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[dict[int, float], list[tuple[int, bool]]]:
    """モデルのbus確率と正解ラベルをバッチ推論で返す。"""
    from compare_models import load_ft_model

    model, class_names = load_ft_model(model_path, classes_path)
    model = model.to(device)
    bus_index = class_names.index("bus")
    scores: dict[int, float] = {}
    labeled_data: list[tuple[int, bool]] = []
    offset = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            probabilities = torch.softmax(model(inputs.to(device)), dim=1)[:, bus_index]
            for local_index, (score, label) in enumerate(
                zip(probabilities.cpu().tolist(), labels.tolist())
            ):
                global_index = offset + local_index
                scores[global_index] = float(score)
                labeled_data.append((global_index, bool(label)))
            offset += len(labels)
    return scores, labeled_data


def evaluate_model(
    name: str,
    model_path: Path,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float | str | int | None]:
    """1モデルについてheld-out testのAP・最大F1・0.5 accuracyを返す。"""
    from compare_models import key_metrics
    from evaluate import compute_average_precision

    if not model_path.is_file():
        raise FileNotFoundError(f"モデルがありません: {model_path}")
    classes_path = model_path.with_name(f"{model_path.stem}_classes.json")
    if name == "synthetic_only" and not classes_path.exists():
        classes_path = model_path.with_name("best_resnet18_bus_classes.json")
    if not classes_path.is_file():
        raise FileNotFoundError(f"クラス定義がありません: {classes_path}")
    scores, labeled_data = score_model(
        model_path, classes_path, dataloader, device
    )
    average_precision = compute_average_precision(scores, labeled_data)
    metrics = key_metrics(scores, labeled_data)
    correct = sum(
        1 for index, label in labeled_data if (scores[index] >= 0.5) == label
    )
    return {
        "model": name,
        "model_path": str(model_path),
        "n": len(labeled_data),
        "average_precision": average_precision,
        "best_f1": metrics["best_f1"],
        "best_f1_threshold": metrics["best_f1_threshold"],
        "recall_at_precision_1": metrics["recall_at_p1"],
        "precision_at_recall_1": metrics["precision_at_r1"],
        "accuracy_at_0_5": correct / len(labeled_data),
    }


def write_results(
    results: list[dict[str, float | str | int | None]],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    """比較結果をCSVとMarkdownへ保存する。"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "real_training_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)

    counts = Counter(row["label"] for row in rows)
    markdown_path = args.output_dir / "real_training_comparison.md"
    lines = [
        "# real_recaptcha学習利用の比較",
        "",
        f"- 固定manifest: `{args.manifest}`",
        f"- manifest SHA-256: `{hashlib.sha256(args.manifest.read_bytes()).hexdigest()}`",
        f"- 評価対象: test splitのみ（bus={counts['bus']} / nonbus={counts['nonbus']}）",
        "- train/valと同じ近重複groupは含まれないことを事前監査済み。",
        "",
        "| model | AP | best F1 | threshold | accuracy@0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['model']} | {result['average_precision']:.4f} | "
            f"{result['best_f1']:.4f} | {result['best_f1_threshold']:.4f} | "
            f"{result['accuracy_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 読み方",
            "",
            "`mixed_real` が `synthetic_only` をheld-out testで上回れば、",
            "本物画像の一部を学習へ加えた効果が、評価リークなしで確認できる。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"csv={csv_path}")
    print(f"markdown={markdown_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合成のみFTと本物混合FTを固定test splitで比較します。"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--real-data-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--baseline-model", type=Path, default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--mixed-model", type=Path, default=DEFAULT_MIXED_MODEL)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--audit-only", action="store_true", help="splitと画像だけ検査し推論しない。"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows = load_and_audit_manifest(args.manifest)
    test_rows = select_test_rows(all_rows, args.max_per_class, args.seed)
    counts = Counter(row["label"] for row in test_rows)
    if not test_rows or counts["bus"] != counts["nonbus"]:
        raise ValueError(f"test splitが空または不均衡です: {dict(counts)}")
    missing = [
        row["image_path"]
        for row in test_rows
        if not (args.real_data_dir / row["image_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"test画像がありません: {missing[:5]}")
    print(
        f"manifest audit OK: test bus={counts['bus']} nonbus={counts['nonbus']}"
    )
    if args.audit_only:
        print("[audit-only] モデル推論は実行していません。")
        return

    device = resolve_device(args.device)
    dataset = EvaluationDataset(test_rows, args.real_data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    results = [
        evaluate_model("synthetic_only", args.baseline_model, dataloader, device),
        evaluate_model("mixed_real", args.mixed_model, dataloader, device),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    write_results(results, test_rows, args)


if __name__ == "__main__":
    main()
