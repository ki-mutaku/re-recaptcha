import argparse
import csv
import hashlib
import random
import shutil
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import yaml


VARIANTS = ("fog", "mosaic")
SPLITS = ("train", "val", "test")
SourcePair = tuple[Path, Path]
SourceGroups = dict[str, dict[str, SourcePair]]


def collect_source_groups(
    base_dir: Path,
    variants: Iterable[str] = VARIANTS,
) -> SourceGroups:
    """元画像IDごとに、各劣化画像と対応ラベルを一つのグループへまとめる。"""
    variant_names = tuple(variants)
    groups: SourceGroups = {}
    missing_directories: list[Path] = []
    missing_labels: list[Path] = []
    orphan_labels: list[Path] = []

    for variant in variant_names:
        image_dir = base_dir / f"test_images_{variant}"
        label_dir = base_dir / f"labels_{variant}"

        for directory in (image_dir, label_dir):
            if not directory.is_dir():
                missing_directories.append(directory)

        if not image_dir.is_dir() or not label_dir.is_dir():
            continue

        image_paths = sorted(image_dir.glob("*.jpg"))
        label_paths = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        orphan_labels.extend(
            path for path in label_paths if path.stem not in image_stems
        )

        for image_path in image_paths:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                missing_labels.append(label_path)
                continue

            groups.setdefault(image_path.stem, {})[variant] = (
                image_path,
                label_path,
            )

    if missing_directories:
        paths = "\n".join(f"- {path}" for path in missing_directories)
        raise FileNotFoundError(
            "セグメンテーション用の入力ディレクトリが不足しています。"
            "add_noise.py と extract_and_apply_labels.py を先に実行してください。\n"
            f"{paths}"
        )

    if missing_labels:
        examples = "\n".join(f"- {path}" for path in missing_labels[:10])
        raise ValueError(
            f"画像に対応するラベルが不足しています ({len(missing_labels)}件)。\n"
            f"{examples}"
        )

    if orphan_labels:
        examples = "\n".join(f"- {path}" for path in orphan_labels[:10])
        raise ValueError(
            f"対応する画像がないラベルがあります ({len(orphan_labels)}件)。\n"
            f"{examples}"
        )

    required_variants = set(variant_names)
    incomplete_groups = {
        source_id: required_variants - set(group)
        for source_id, group in groups.items()
        if set(group) != required_variants
    }
    if incomplete_groups:
        examples = "\n".join(
            f"- {source_id}: {', '.join(sorted(missing))}"
            for source_id, missing in list(incomplete_groups.items())[:10]
        )
        raise ValueError(
            "同じ元画像IDの劣化バリエーションが不足しています "
            f"({len(incomplete_groups)}件)。\n{examples}"
        )

    if not groups:
        raise ValueError("有効な画像とラベルのグループが見つかりませんでした。")

    return groups


def split_source_ids(
    source_ids: Mapping[str, object] | Iterable[str],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """同じ元画像IDを必ず同じsplitへ割り当て、データリークを防ぐ。"""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio は 0 より大きく 1 より小さくしてください。")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio は 0 より大きく 1 より小さくしてください。")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio は 1 より小さくしてください。")

    shuffled_ids = sorted(source_ids)
    random.Random(seed).shuffle(shuffled_ids)

    train_count = int(len(shuffled_ids) * train_ratio)
    val_count = int(len(shuffled_ids) * val_ratio)
    test_count = len(shuffled_ids) - train_count - val_count
    if min(train_count, val_count, test_count) == 0:
        raise ValueError(
            "train/val/test の各splitに1件以上必要です。"
            "データ数または分割比率を見直してください。"
        )

    assignments: dict[str, str] = {}
    for source_id in shuffled_ids[:train_count]:
        assignments[source_id] = "train"
    for source_id in shuffled_ids[train_count : train_count + val_count]:
        assignments[source_id] = "val"
    for source_id in shuffled_ids[train_count + val_count :]:
        assignments[source_id] = "test"

    return assignments


def write_split_manifest(
    manifest_path: Path,
    assignments: Mapping[str, str],
    groups: SourceGroups,
) -> None:
    """splitと入力内容の指紋を保存し、再生成時の意図しない変更を検出可能にする。"""
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("source_id", "split", "fingerprint"),
        )
        writer.writeheader()
        for source_id in sorted(assignments):
            writer.writerow(
                {
                    "source_id": source_id,
                    "split": assignments[source_id],
                    "fingerprint": compute_source_fingerprint(groups[source_id]),
                }
            )


def compute_source_fingerprint(group: Mapping[str, SourcePair]) -> str:
    """同じ元画像IDに属する全画像・ラベル内容から再現性確認用の指紋を作る。"""
    digest = hashlib.sha256()
    for variant in VARIANTS:
        for kind, path in zip(("image", "label"), group[variant], strict=True):
            digest.update(f"{variant}:{kind}\0".encode())
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def read_split_manifest(
    manifest_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """既存manifestからsplitと入力指紋を読み込み、再生成時の比較に使う。"""
    assignments: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8", newline="") as manifest:
        reader = csv.DictReader(manifest)
        if reader.fieldnames != ["source_id", "split", "fingerprint"]:
            raise ValueError(
                f"split manifestの列が不正です: {manifest_path}"
            )

        for row in reader:
            source_id = row["source_id"]
            split = row["split"]
            if source_id in assignments:
                raise ValueError(
                    f"split manifestに元画像IDが重複しています: {source_id}"
                )
            if split not in SPLITS:
                raise ValueError(
                    f"split manifestに不正なsplitがあります: {source_id}={split}"
                )
            assignments[source_id] = split
            fingerprints[source_id] = row["fingerprint"]

    return assignments, fingerprints


def copy_grouped_data(
    groups: SourceGroups,
    assignments: Mapping[str, str],
    dataset_dir: Path,
) -> None:
    """元画像IDの割り当てに従い、全バリエーションを同じsplitへコピーする。"""
    for split in SPLITS:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for source_id in sorted(groups):
        split = assignments[source_id]
        for variant in VARIANTS:
            image_path, label_path = groups[source_id][variant]
            output_stem = f"{variant}_{source_id}"
            shutil.copy2(
                image_path,
                dataset_dir / "images" / split / f"{output_stem}{image_path.suffix}",
            )
            shutil.copy2(
                label_path,
                dataset_dir / "labels" / split / f"{output_stem}.txt",
            )


def validate_dataset(
    dataset_dir: Path,
    assignments: Mapping[str, str],
) -> None:
    """画像・ラベルの欠落、余分なファイル、元画像IDのsplit重複を検出する。"""
    actual_ids: dict[str, set[str]] = {split: set() for split in SPLITS}

    for split in SPLITS:
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))

        expected_count = sum(
            len(VARIANTS)
            for assigned_split in assignments.values()
            if assigned_split == split
        )
        if len(images) != expected_count or len(labels) != expected_count:
            raise ValueError(
                f"{split} のファイル数が不正です: "
                f"images={len(images)}, labels={len(labels)}, expected={expected_count}"
            )

        for image_path in images:
            variant, separator, source_id = image_path.stem.partition("_")
            if not separator or variant not in VARIANTS:
                raise ValueError(f"不正な画像ファイル名です: {image_path.name}")
            if assignments.get(source_id) != split:
                raise ValueError(
                    f"manifestと異なるsplitに画像があります: {image_path.name}"
                )
            if not (label_dir / f"{image_path.stem}.txt").is_file():
                raise ValueError(f"対応ラベルがありません: {image_path.name}")
            actual_ids[split].add(source_id)

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1 :]:
            overlap = actual_ids[first_split] & actual_ids[second_split]
            if overlap:
                examples = ", ".join(sorted(overlap)[:10])
                raise ValueError(
                    f"データリークを検出しました: {first_split}/{second_split}: "
                    f"{examples}"
                )


def replace_dataset(staging_dir: Path, dataset_dir: Path) -> Path | None:
    """検証済みデータへ入れ替え、既存データセットは日時付きで退避する。"""
    backup_dir = None
    if dataset_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup_dir = dataset_dir.with_name(f"{dataset_dir.name}.backup-{timestamp}")
        dataset_dir.rename(backup_dir)

    try:
        staging_dir.rename(dataset_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not dataset_dir.exists():
            backup_dir.rename(dataset_dir)
        raise

    return backup_dir


def resolve_dataset_dir(base_dir: Path, dataset_dir: Path | None) -> Path:
    """明示された相対データセットパスも、他の生成物と同じsrc基準で解決する。"""
    if dataset_dir is None:
        return base_dir / "yolo_dataset"
    if dataset_dir.is_absolute():
        return dataset_dir
    return base_dir / dataset_dir


def build_dataset(
    *,
    base_dir: Path,
    dataset_dir: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    reuse_manifest: bool = True,
) -> Path:
    """リークのないYOLOデータセットを一時領域で構築し、検証後に配置する。"""
    dataset_dir = resolve_dataset_dir(base_dir, dataset_dir)
    staging_dir = dataset_dir.with_name(f".{dataset_dir.name}.staging")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    groups = collect_source_groups(base_dir)
    existing_manifest = dataset_dir / "split_manifest.csv"
    if reuse_manifest and existing_manifest.is_file():
        assignments, expected_fingerprints = read_split_manifest(existing_manifest)
        source_ids = set(groups)
        manifest_ids = set(assignments)
        if source_ids != manifest_ids:
            added = sorted(source_ids - manifest_ids)
            removed = sorted(manifest_ids - source_ids)
            raise ValueError(
                "入力データと既存split manifestの元画像IDが一致しません。"
                "意図的に再分割する場合は --reshuffle を指定してください。"
                f" added={len(added)}, removed={len(removed)}"
            )
        changed_ids = [
            source_id
            for source_id in sorted(source_ids)
            if expected_fingerprints[source_id]
            != compute_source_fingerprint(groups[source_id])
        ]
        if changed_ids:
            examples = ", ".join(changed_ids[:10])
            raise ValueError(
                "既存split manifest作成後に入力画像またはラベルの内容が"
                "変更されています。意図的に再分割する場合は --reshuffle を"
                f"指定してください。 changed={len(changed_ids)}: {examples}"
            )
        print(f"既存のsplit割り当てを再利用します: {existing_manifest}")
    else:
        assignments = split_source_ids(
            groups,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

    try:
        copy_grouped_data(groups, assignments, staging_dir)
        write_split_manifest(
            staging_dir / "split_manifest.csv",
            assignments,
            groups,
        )
        validate_dataset(staging_dir, assignments)
        backup_dir = replace_dataset(staging_dir, dataset_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    split_counts = {
        split: sum(assigned == split for assigned in assignments.values())
        for split in SPLITS
    }
    print(
        "データセットを元画像ID単位で作成しました: "
        + ", ".join(f"{split}={count}件" for split, count in split_counts.items())
    )
    if backup_dir is not None:
        print(f"既存データセットは {backup_dir} に退避しました。")

    return dataset_dir


def setup_dataset(
    *,
    dataset_dir: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    reuse_manifest: bool = True,
) -> Path:
    """src直下の生成データから学習・検証・テスト用データセットを作る。"""
    base_dir = Path(__file__).resolve().parents[1]
    return build_dataset(
        base_dir=base_dir,
        dataset_dir=dataset_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        reuse_manifest=reuse_manifest,
    )


def create_yaml(dataset_dir: Path, model: object) -> Path:
    """YOLOが各splitとCOCOクラス名を読み込むためのdata.yamlを生成する。"""
    yaml_path = dataset_dir / "data.yaml"
    class_names = model.names
    yaml_data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }

    with yaml_path.open("w", encoding="utf-8") as yaml_file:
        yaml.dump(yaml_data, yaml_file, sort_keys=False, allow_unicode=True)

    print(f"{yaml_path} を作成しました。")
    return yaml_path


def parse_args() -> argparse.Namespace:
    """ローカル実行とSlurm実行で同じ学習設定を指定できるCLIを提供する。"""
    parser = argparse.ArgumentParser(
        description="霧・モザイク画像でYOLOv8 segmentationを学習します。"
    )
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="yolov8_segmentation_runs")
    parser.add_argument("--name", default="fog_mosaic_finetune")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--reshuffle",
        action="store_true",
        help="既存manifestを使わず、指定した比率とseedで再分割します。",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="データセットとdata.yamlだけ作成し、学習は開始しません。",
    )
    return parser.parse_args()


def main() -> None:
    """データセットの再構築、設定生成、YOLOv8-segの学習を順番に実行する。"""
    args = parse_args()
    dataset_dir = setup_dataset(
        dataset_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        reuse_manifest=not args.reshuffle,
    )

    from ultralytics import YOLO

    print(f"モデル {args.model} をロードしています...")
    model = YOLO(args.model)
    yaml_path = create_yaml(dataset_dir, model)

    if args.prepare_only:
        print("データセットの準備が完了しました。学習は開始していません。")
        return

    print("ファインチューニングを開始します...")
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=None if args.device == "auto" else args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )
    print("学習が完了しました！")


if __name__ == "__main__":
    main()
