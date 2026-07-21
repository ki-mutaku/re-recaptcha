import csv
import tempfile
import unittest
from pathlib import Path

from src.segmentation.train_yolov8_segmentation import (
    build_dataset,
    collect_source_groups,
    resolve_dataset_dir,
    split_source_ids,
)


class SegmentationDatasetTest(unittest.TestCase):
    """同じ元画像の派生画像が split をまたがないことを保証する。"""

    def setUp(self):
        """各テストを実データから隔離するため、10元画像分の一時入力を作る。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        for variant in ("fog", "mosaic"):
            image_dir = self.base_dir / f"test_images_{variant}"
            label_dir = self.base_dir / f"labels_{variant}"
            image_dir.mkdir()
            label_dir.mkdir()

            for index in range(10):
                source_id = f"{index:012d}"
                (image_dir / f"{source_id}.jpg").write_bytes(
                    f"{variant}-{source_id}".encode()
                )
                (label_dir / f"{source_id}.txt").write_text(
                    "5 0.1 0.1 0.2 0.1 0.2 0.2\n",
                    encoding="utf-8",
                )

    def tearDown(self):
        """テストで作った画像・ラベル・バックアップを一時領域ごと破棄する。"""
        self.temp_dir.cleanup()

    def test_split_source_ids_assigns_each_source_to_exactly_one_split(self):
        """各元画像IDがtrain/val/testのどれか一つだけに割り当てられることを確認する。"""
        groups = collect_source_groups(self.base_dir)

        assignments = split_source_ids(
            groups,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )

        self.assertEqual(len(assignments), 10)
        self.assertEqual(set(assignments.values()), {"train", "val", "test"})
        self.assertEqual(sum(split == "train" for split in assignments.values()), 8)
        self.assertEqual(sum(split == "val" for split in assignments.values()), 1)
        self.assertEqual(sum(split == "test" for split in assignments.values()), 1)

    def test_build_dataset_keeps_variants_together_and_removes_stale_files(self):
        """fog/mosaicを同じsplitに保ち、古いファイルを新データへ混ぜないことを確認する。"""
        dataset_dir = self.base_dir / "yolo_dataset"
        stale_file = dataset_dir / "images" / "train" / "stale.jpg"
        stale_file.parent.mkdir(parents=True)
        stale_file.write_bytes(b"stale")

        result = build_dataset(
            base_dir=self.base_dir,
            dataset_dir=dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )

        self.assertEqual(result, dataset_dir)
        self.assertFalse(stale_file.exists())
        backups = list(self.base_dir.glob("yolo_dataset.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertTrue((backups[0] / "images" / "train" / "stale.jpg").exists())

        source_splits: dict[str, set[str]] = {}
        for split in ("train", "val", "test"):
            image_dir = dataset_dir / "images" / split
            label_dir = dataset_dir / "labels" / split

            for image_path in image_dir.glob("*.jpg"):
                variant, source_id = image_path.stem.split("_", maxsplit=1)
                source_splits.setdefault(source_id, set()).add(split)
                self.assertIn(variant, {"fog", "mosaic"})
                self.assertTrue((label_dir / f"{image_path.stem}.txt").exists())

        self.assertEqual(len(source_splits), 10)
        self.assertTrue(all(len(splits) == 1 for splits in source_splits.values()))

        with (dataset_dir / "split_manifest.csv").open(newline="") as manifest:
            rows = list(csv.DictReader(manifest))

        self.assertEqual(len(rows), 10)
        self.assertEqual({row["source_id"] for row in rows}, set(source_splits))

    def test_build_dataset_reuses_existing_manifest(self):
        """再実行時はseedや比率より既存manifestを優先することを確認する。"""
        dataset_dir = self.base_dir / "yolo_dataset"
        build_dataset(
            base_dir=self.base_dir,
            dataset_dir=dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )
        original_manifest = (dataset_dir / "split_manifest.csv").read_text(
            encoding="utf-8"
        )

        build_dataset(
            base_dir=self.base_dir,
            dataset_dir=dataset_dir,
            train_ratio=0.6,
            val_ratio=0.2,
            seed=999,
        )

        self.assertEqual(
            (dataset_dir / "split_manifest.csv").read_text(encoding="utf-8"),
            original_manifest,
        )

    def test_build_dataset_rejects_content_drift(self):
        """同じIDでも画像内容が変わった場合はmanifest再利用を中断することを確認する。"""
        dataset_dir = self.base_dir / "yolo_dataset"
        build_dataset(base_dir=self.base_dir, dataset_dir=dataset_dir)
        changed_image = self.base_dir / "test_images_fog" / "000000000000.jpg"
        changed_image.write_bytes(b"changed")

        with self.assertRaisesRegex(ValueError, "内容が変更"):
            build_dataset(base_dir=self.base_dir, dataset_dir=dataset_dir)

    def test_collect_source_groups_rejects_missing_labels(self):
        """画像に対応するラベルの欠落を、コピー開始前に検出することを確認する。"""
        missing_label = self.base_dir / "labels_fog" / "000000000000.txt"
        missing_label.unlink()

        with self.assertRaisesRegex(ValueError, "ラベルが不足"):
            collect_source_groups(self.base_dir)

    def test_collect_source_groups_rejects_orphan_labels(self):
        """対応画像がないラベルを、データセット構築前に検出することを確認する。"""
        orphan_label = self.base_dir / "labels_fog" / "orphan.txt"
        orphan_label.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "対応する画像がないラベル"):
            collect_source_groups(self.base_dir)

    def test_resolve_dataset_dir_uses_base_dir_for_relative_paths(self):
        """CLIの相対データセット指定もsrc基準として解決されることを確認する。"""
        self.assertEqual(
            resolve_dataset_dir(self.base_dir, Path("custom_dataset")),
            self.base_dir / "custom_dataset",
        )


if __name__ == "__main__":
    unittest.main()
