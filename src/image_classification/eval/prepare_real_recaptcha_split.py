"""real_recaptcha を近重複グループ単位で固定分割する。

画像そのものはgit管理しないが、このスクリプトが作るCSVとメタデータはgitで共有する。
同一画像（SHA-256一致）と見た目が近い画像（dHashのHamming距離が閾値以下）を
同じ group_id に束ね、同じグループが train/val/test をまたがないようにする。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent
DEFAULT_DATA_DIR = PKG_DIR / "data" / "real_recaptcha"
DEFAULT_OUTPUT = HERE / "splits" / "real_recaptcha_split.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ImageRecord:
    """分割に必要な画像1枚分の不変メタデータ。"""

    relative_path: str
    label: str
    has_bus: bool
    sha256: str
    dhash: int


class UnionFind:
    """重複関係の推移閉包を効率よくグループ化する。"""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def calculate_dhash(path: Path) -> int:
    """100x100画像の見た目の近さを比較する64bit difference hashを返す。"""
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def scan_images(data_dir: Path) -> list[ImageRecord]:
    """bus/nonbus配下を走査し、内容ハッシュと知覚ハッシュを計算する。"""
    records: list[ImageRecord] = []
    for label, has_bus in (("bus", True), ("nonbus", False)):
        label_dir = data_dir / label
        if not label_dir.is_dir():
            raise FileNotFoundError(f"画像ディレクトリがありません: {label_dir}")
        paths = sorted(
            path for path in label_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for index, path in enumerate(paths, start=1):
            raw = path.read_bytes()
            records.append(
                ImageRecord(
                    relative_path=path.relative_to(data_dir).as_posix(),
                    label=label,
                    has_bus=has_bus,
                    sha256=hashlib.sha256(raw).hexdigest(),
                    dhash=calculate_dhash(path),
                )
            )
            if index % 1000 == 0:
                print(f"[hash] {label}: {index}/{len(paths)}")
    if not records:
        raise RuntimeError(f"画像がありません: {data_dir}")
    return records


def build_duplicate_groups(
    records: list[ImageRecord], near_duplicate_distance: int
) -> dict[int, list[int]]:
    """完全一致と近重複をunionし、rootから画像index群への辞書を返す。"""
    union_find = UnionFind(len(records))

    exact_hashes: dict[str, int] = {}
    for index, record in enumerate(records):
        previous = exact_hashes.setdefault(record.sha256, index)
        union_find.union(previous, index)

    # 8bitごとのLSHバンドを使い、Hamming距離が近い候補だけを比較する。
    # 距離4以下なら8バンド中少なくとも4バンドが一致するため候補漏れがない。
    representatives: dict[int, int] = {}
    for index, record in enumerate(records):
        representative = representatives.setdefault(record.dhash, index)
        # raw bytesが異なってもdHashが同一なら距離0の近重複として扱う。
        union_find.union(representative, index)

    buckets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for dhash, index in representatives.items():
        for band in range(8):
            buckets[(band, (dhash >> (band * 8)) & 0xFF)].append((dhash, index))

    compared: set[tuple[int, int]] = set()
    for candidates in buckets.values():
        for left_pos, (left_hash, left_index) in enumerate(candidates):
            for right_hash, right_index in candidates[left_pos + 1 :]:
                pair = (
                    min(left_hash, right_hash),
                    max(left_hash, right_hash),
                )
                if pair in compared:
                    continue
                compared.add(pair)
                if (left_hash ^ right_hash).bit_count() <= near_duplicate_distance:
                    union_find.union(left_index, right_index)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union_find.find(index)].append(index)
    return dict(groups)


def assign_groups(
    records: list[ImageRecord],
    groups: dict[int, list[int]],
    ratios: dict[str, float],
    seed: int,
) -> dict[int, str]:
    """クラス比を保ちながら、グループ単位で決定的にsplitを割り当てる。"""
    totals = Counter(record.label for record in records)
    targets = {
        split: {label: totals[label] * ratios[split] for label in totals}
        for split in SPLITS
    }
    assigned = {split: Counter() for split in SPLITS}

    def group_order(item: tuple[int, list[int]]) -> tuple[int, str]:
        _, members = item
        signature = "|".join(sorted(records[index].sha256 for index in members))
        digest = hashlib.sha256(f"{seed}|{signature}".encode()).hexdigest()
        return (-len(members), digest)

    assignments: dict[int, str] = {}
    for root, members in sorted(groups.items(), key=group_order):
        counts = Counter(records[index].label for index in members)

        def score(split: str) -> tuple[float, str]:
            benefit = 0.0
            overflow = 0.0
            for label, count in counts.items():
                target = max(targets[split][label], 1.0)
                remaining = target - assigned[split][label]
                benefit += min(max(remaining, 0.0), count) / target
                overflow += max(count - max(remaining, 0.0), 0.0) / target
            tie = hashlib.sha256(f"{seed}|{root}|{split}".encode()).hexdigest()
            return (benefit - overflow, tie)

        chosen = max(SPLITS, key=score)
        assignments[root] = chosen
        assigned[chosen].update(counts)
    return assignments


def validate_manifest_rows(rows: list[dict[str, str]]) -> None:
    """パス重複とgroupのsplitまたぎがないことを検査する。"""
    paths = [row["image_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("manifest内にimage_pathの重複があります")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"近重複グループがsplitをまたいでいます: {leaking[:5]}")


def audit_existing_manifest(path: Path, data_dir: Path) -> None:
    """既存の固定manifestを変更せず、構造とローカル画像の存在を検査する。"""
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    validate_manifest_rows(rows)
    missing = [row["image_path"] for row in rows if not (data_dir / row["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"manifestに対応する画像がありません: {missing[:5]}")
    counts = Counter((row["split"], row["label"]) for row in rows)
    print(f"[audit] manifest={path} rows={len(rows)}")
    for split in SPLITS:
        print(
            f"  {split}: bus={counts[(split, 'bus')]} "
            f"nonbus={counts[(split, 'nonbus')]}"
        )


def write_manifest(
    records: list[ImageRecord],
    groups: dict[int, list[int]],
    assignments: dict[int, str],
    output_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
) -> None:
    """固定CSVと再現条件を記録したJSONを原子的に近い順序で保存する。"""
    root_by_index = {
        index: root for root, members in groups.items() for index in members
    }
    ordered_roots = sorted(
        groups,
        key=lambda root: min(records[index].sha256 for index in groups[root]),
    )
    group_ids = {root: f"g{position:05d}" for position, root in enumerate(ordered_roots)}

    rows = []
    for index, record in enumerate(records):
        root = root_by_index[index]
        rows.append(
            {
                "image_path": record.relative_path,
                "label": record.label,
                "has_bus": str(record.has_bus).lower(),
                "split": assignments[root],
                "sha256": record.sha256,
                "dhash": f"{record.dhash:016x}",
                "group_id": group_ids[root],
            }
        )
    rows.sort(key=lambda row: row["image_path"])
    validate_manifest_rows(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    dataset_fingerprint = hashlib.sha256(
        "\n".join(f"{row['image_path']}|{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    manifest_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    counts = Counter((row["split"], row["label"]) for row in rows)
    mixed_label_groups = sum(
        1
        for members in groups.values()
        if len({records[index].label for index in members}) > 1
    )
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 10),
        },
        "near_duplicate_hamming_distance": args.near_duplicate_distance,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "manifest_sha256": manifest_sha256,
        "images": len(records),
        "groups": len(groups),
        "mixed_label_groups": mixed_label_groups,
        "counts": {
            split: {
                label: counts[(split, label)] for label in ("bus", "nonbus")
            }
            for split in SPLITS
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="real_recaptchaを近重複グループ単位で固定分割します。"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument(
        "--force",
        action="store_true",
        help="既存manifestを上書きする。通常は固定split保護のため指定しない。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise SystemExit("train-ratioとval-ratioは0より大きくしてください")
    if args.train_ratio + args.val_ratio >= 1:
        raise SystemExit("train-ratio + val-ratio は1未満にしてください")
    if not 0 <= args.near_duplicate_distance <= 7:
        raise SystemExit("near-duplicate-distanceは0〜7を指定してください")

    output_path = args.output.resolve()
    data_dir = args.data_dir.resolve()
    metadata_path = output_path.with_suffix(".meta.json")
    if output_path.exists() and not args.force:
        audit_existing_manifest(output_path, data_dir)
        print("既存manifestを固定したまま使用します。再分割する場合だけ --force を指定。")
        return

    records = scan_images(data_dir)
    groups = build_duplicate_groups(records, args.near_duplicate_distance)
    test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 10)
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": test_ratio,
    }
    assignments = assign_groups(records, groups, ratios, args.seed)
    write_manifest(records, groups, assignments, output_path, metadata_path, args)
    audit_existing_manifest(output_path, data_dir)


if __name__ == "__main__":
    main()
