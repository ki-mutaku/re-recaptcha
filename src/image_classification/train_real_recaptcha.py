"""固定splitのreal_recaptchaを一部だけ学習へ加え、別名モデルを作る。

既存の合成データのみモデル ``best_resnet18_bus.pth`` は上書きしない。
既定では合成trainと、本物trainから各600枚を混ぜ、本物val各300枚だけで
ベストepochを選ぶ。test splitはこのスクリプトから一切読み込まない。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import datasets, models, transforms

HERE = Path(__file__).resolve().parent
DEFAULT_SYNTHETIC_DIR = HERE / "data" / "dataset"
DEFAULT_REAL_DIR = HERE / "data" / "real_recaptcha"
DEFAULT_MANIFEST = HERE / "eval" / "splits" / "real_recaptcha_split.csv"
DEFAULT_MODEL = HERE / "models" / "best_resnet18_bus_mixed_real.pth"
CLASS_NAMES = ["bus", "other"]
LABEL_TO_INDEX = {"bus": 0, "nonbus": 1}


class ManifestImageDataset(Dataset):
    """manifestのパスと二値ラベルだけを読む、コピー不要の画像Dataset。"""

    def __init__(
        self,
        rows: list[dict[str, str]],
        data_dir: Path,
        transform: transforms.Compose,
    ) -> None:
        self.rows = rows
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(self.data_dir / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, LABEL_TO_INDEX[row["label"]]


def set_seed(seed: int) -> None:
    """Python・NumPy・PyTorchの乱数を固定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """auto指定をCUDA、MPS、CPUの優先順で実デバイスへ解決する。"""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> list[dict[str, str]]:
    """固定manifestを読み、testが他splitと同じgroupを共有しないことを確認する。"""
    if not path.is_file():
        raise FileNotFoundError(
            f"固定splitがありません: {path}\n"
            "先に eval/prepare_real_recaptcha_split.py を実行してください。"
        )
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image_path", "label", "split", "sha256", "group_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifestの列が不足しています: {required}")
    invalid_labels = sorted({row["label"] for row in rows} - set(LABEL_TO_INDEX))
    invalid_splits = sorted({row["split"] for row in rows} - {"train", "val", "test"})
    if invalid_labels:
        raise ValueError(f"manifestに未知のlabelがあります: {invalid_labels}")
    if invalid_splits:
        raise ValueError(f"manifestに未知のsplitがあります: {invalid_splits}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leaking = [group for group, split_set in group_splits.items() if len(split_set) > 1]
    if leaking:
        raise ValueError(f"groupがsplitをまたいでいます: {leaking[:5]}")
    return rows


def stable_subset(
    rows: list[dict[str, str]], split: str, per_class: int, seed: int
) -> list[dict[str, str]]:
    """指定splitから各クラス同数をmanifest内容に基づき決定的に選ぶ。"""
    selected: list[dict[str, str]] = []
    for label in ("bus", "nonbus"):
        candidates = [
            row for row in rows if row["split"] == split and row["label"] == label
        ]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{split}|{row['sha256']}".encode()
            ).hexdigest()
        )
        if per_class > 0:
            candidates = candidates[:per_class]
        selected.extend(candidates)
    selected.sort(key=lambda row: row["image_path"])
    return selected


def make_transforms() -> dict[str, transforms.Compose]:
    """既存train_resnet.pyと同じ学習・検証前処理を返す。"""
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        ),
    }


def build_datasets(
    args: argparse.Namespace, rows: list[dict[str, str]]
) -> tuple[Dataset, Dataset, dict[str, int]]:
    """合成＋本物trainと、本物だけのval Datasetを構築する。"""
    data_transforms = make_transforms()
    real_train_rows = stable_subset(
        rows, "train", args.real_train_per_class, args.seed
    )
    real_val_rows = stable_subset(rows, "val", args.real_val_per_class, args.seed)

    expected_train = args.real_train_per_class * len(LABEL_TO_INDEX)
    expected_val = args.real_val_per_class * len(LABEL_TO_INDEX)
    if args.real_train_per_class > 0 and len(real_train_rows) != expected_train:
        raise ValueError(
            "real trainの枚数が不足しています: "
            f"requested={expected_train} selected={len(real_train_rows)}"
        )
    if args.real_val_per_class > 0 and len(real_val_rows) != expected_val:
        raise ValueError(
            "real valの枚数が不足しています: "
            f"requested={expected_val} selected={len(real_val_rows)}"
        )
    if any(row["split"] == "test" for row in real_train_rows + real_val_rows):
        raise AssertionError("学習・validation選択にtestが混入しました")

    missing = [
        row["image_path"]
        for row in real_train_rows + real_val_rows
        if not (args.real_data_dir / row["image_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"本物画像がありません: {missing[:5]}")

    real_train = ManifestImageDataset(
        real_train_rows, args.real_data_dir, data_transforms["train"]
    )
    real_val = ManifestImageDataset(
        real_val_rows, args.real_data_dir, data_transforms["val"]
    )

    train_parts: list[Dataset] = [real_train]
    synthetic_count = 0
    if args.mode == "mixed":
        synthetic_train_dir = args.synthetic_data_dir / "train"
        if not synthetic_train_dir.is_dir():
            raise FileNotFoundError(f"合成trainがありません: {synthetic_train_dir}")
        synthetic_train = datasets.ImageFolder(
            synthetic_train_dir, data_transforms["train"]
        )
        if synthetic_train.classes != CLASS_NAMES:
            raise ValueError(
                f"合成データのクラス順が想定外です: {synthetic_train.classes}"
            )
        train_parts.insert(0, synthetic_train)
        synthetic_count = len(synthetic_train)

    train_dataset: Dataset
    if len(train_parts) == 1:
        train_dataset = train_parts[0]
    else:
        train_dataset = ConcatDataset(train_parts)
    counts = {
        "synthetic_train": synthetic_count,
        "real_train": len(real_train),
        "real_val": len(real_val),
    }
    return train_dataset, real_val, counts


def calculate_f1(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """bus=0を陽性としてprecision・recall・F1を返す。"""
    true_positive = int(((predictions == 0) & (labels == 0)).sum())
    false_positive = int(((predictions == 0) & (labels != 0)).sum())
    false_negative = int(((predictions != 0) & (labels == 0)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def train_model(
    args: argparse.Namespace,
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    """本物valのbus F1が最大のResNet18重みを返す。"""
    generator = torch.Generator().manual_seed(args.seed)
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        ),
    }
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_weights = copy.deepcopy(model.state_dict())
    best_metrics = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "epoch": 0}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for phase in ("train", "val"):
            model.train(phase == "train")
            total_loss = 0.0
            predictions_all: list[torch.Tensor] = []
            labels_all: list[torch.Tensor] = []
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                predictions_all.append(outputs.argmax(dim=1).detach().cpu())
                labels_all.append(labels.detach().cpu())

            predictions = torch.cat(predictions_all)
            label_values = torch.cat(labels_all)
            accuracy = float((predictions == label_values).float().mean())
            metrics = calculate_f1(predictions, label_values)
            average_loss = total_loss / len(dataloaders[phase].dataset)
            print(
                f"{phase}: loss={average_loss:.4f} accuracy={accuracy:.4f} "
                f"bus_f1={metrics['f1']:.4f}"
            )
            if phase == "val" and metrics["f1"] > best_metrics["f1"]:
                best_weights = copy.deepcopy(model.state_dict())
                best_metrics = {**metrics, "accuracy": accuracy, "epoch": epoch}

    model.load_state_dict(best_weights)
    best_metrics["elapsed_seconds"] = time.time() - started
    return model, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="固定splitのreal_recaptchaを一部使って別名FTモデルを学習します。"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--real-data-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--synthetic-data-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--mode", choices=("mixed", "real-only"), default="mixed")
    parser.add_argument("--real-train-per-class", type=int, default=600)
    parser.add_argument("--real-val-per-class", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--dry-run", action="store_true", help="データ検査だけ行い学習しない。"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_train_per_class <= 0 or args.real_val_per_class <= 0:
        raise SystemExit("real-train-per-classとreal-val-per-classは1以上にしてください")
    set_seed(args.seed)
    rows = load_manifest(args.manifest)
    train_dataset, val_dataset, counts = build_datasets(args, rows)
    print(f"mode={args.mode} counts={counts}")
    print("test splitは学習・validationから除外されています。")
    if args.dry_run:
        print("[dry-run] データ検査完了。モデル学習は実行していません。")
        return

    device = resolve_device(args.device)
    print(f"device={device}")
    model, best_metrics = train_model(
        args, train_dataset, val_dataset, device
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_model)
    classes_path = args.output_model.with_name(
        f"{args.output_model.stem}_classes.json"
    )
    classes_path.write_text(
        json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model": str(args.output_model),
        "classes": CLASS_NAMES,
        "mode": args.mode,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "counts": counts,
        "best_validation_metrics": best_metrics,
    }
    metadata_path = args.output_model.with_name(
        f"{args.output_model.stem}_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"model={args.output_model}")
    print(f"classes={classes_path}")
    print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
