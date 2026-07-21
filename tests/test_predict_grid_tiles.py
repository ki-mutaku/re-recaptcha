import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.segmentation.predict_grid_tiles import (
    calculate_tile_metrics,
    compare_prediction_with_expected,
    extract_target_mask,
    increment_output_dir,
    load_yolo_target_mask,
    select_tiles_from_mask,
    tile_bounds,
)


class PredictGridTilesTest(unittest.TestCase):
    def test_tile_bounds_cover_uneven_image_without_gaps(self):
        """4で割り切れない画像でも全画素を16タイルへ割り当てることを確認する。"""
        bounds = tile_bounds(width=7, height=5)

        self.assertEqual(len(bounds), 16)
        self.assertEqual(bounds[0], (1, 0, 0, 1, 1))
        self.assertEqual(bounds[-1], (16, 5, 3, 7, 5))

    def test_select_tiles_uses_one_based_row_major_numbers(self):
        """左上と右下のマスク画素を1番・16番として選択することを確認する。"""
        mask = torch.zeros((8, 8), dtype=torch.bool)
        mask[0, 0] = True
        mask[7, 7] = True

        selected, coverages = select_tiles_from_mask(mask)

        self.assertEqual(selected, [1, 16])
        self.assertGreater(coverages[1], 0.0)
        self.assertGreater(coverages[16], 0.0)

    def test_select_tiles_applies_minimum_mask_ratio(self):
        """指定した最小面積率より小さいマスクを選択対象から除外することを確認する。"""
        mask = torch.zeros((8, 8), dtype=torch.bool)
        mask[0, 0] = True

        selected, _ = select_tiles_from_mask(mask, min_mask_ratio=0.3)

        self.assertEqual(selected, [])

    def test_extract_target_mask_ignores_other_classes(self):
        """複数クラスの予測から指定クラスのマスクだけを統合することを確認する。"""
        masks = torch.zeros((2, 8, 8))
        masks[0, 0, 0] = 1
        masks[1, 7, 7] = 1
        result = SimpleNamespace(
            orig_shape=(8, 8),
            boxes=SimpleNamespace(cls=torch.tensor([5.0, 2.0])),
            masks=SimpleNamespace(data=masks),
        )

        target_mask = extract_target_mask(result, target_class_id=5)
        selected, _ = select_tiles_from_mask(target_mask)

        self.assertEqual(selected, [1])

    def test_load_yolo_target_mask_filters_class_polygons(self):
        """YOLOポリゴンラベルから対象クラスだけの正解マスクを作ることを確認する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "image.txt"
            label_path.write_text(
                "5 0.01 0.01 0.20 0.01 0.20 0.20 0.01 0.20\n"
                "2 0.80 0.80 0.99 0.80 0.99 0.99 0.80 0.99\n",
                encoding="utf-8",
            )

            target_mask = load_yolo_target_mask(
                label_path,
                width=100,
                height=100,
                target_class_id=5,
            )
            selected, _ = select_tiles_from_mask(target_mask)

        self.assertEqual(selected, [1])

    def test_load_yolo_target_mask_keeps_grid_boundary_in_correct_tile(self):
        """x=0.25から始まるポリゴンを境界左側のタイルへ混入させないことを確認する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            label_path = Path(temporary_directory) / "image.txt"
            label_path.write_text(
                "5 0.25 0.01 0.49 0.01 0.49 0.20 0.25 0.20\n",
                encoding="utf-8",
            )

            target_mask = load_yolo_target_mask(
                label_path,
                width=100,
                height=100,
                target_class_id=5,
            )
            selected, _ = select_tiles_from_mask(target_mask)

        self.assertEqual(selected, [2])

    def test_comparison_applies_minimum_ratio_only_to_prediction(self):
        """予測の微小領域だけを除外し、同じ大きさの正解領域は維持することを確認する。"""
        predicted_mask = torch.zeros((8, 8), dtype=torch.bool)
        expected_mask = torch.zeros((8, 8), dtype=torch.bool)
        predicted_mask[0, 0] = True
        expected_mask[0, 0] = True

        predicted, expected, _ = compare_prediction_with_expected(
            predicted_mask,
            expected_mask,
            min_prediction_mask_ratio=0.3,
        )

        self.assertEqual(predicted, [])
        self.assertEqual(expected, [1])

    def test_calculate_tile_metrics_includes_exact_match(self):
        """タイル単位の混同行列と画像単位の完全一致率を計算できることを確認する。"""
        metrics = calculate_tile_metrics(
            predicted_tiles=[{1, 2}, {4}],
            expected_tiles=[{1, 3}, {4}],
            tile_count=16,
        )

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tn"], 28)
        self.assertEqual(metrics["accuracy"], 30 / 32)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 2 / 3)
        self.assertEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["exact_match"], 0.5)

    def test_increment_output_dir_preserves_existing_results(self):
        """既存の予測結果を上書きせず連番付きディレクトリを選ぶことを確認する。"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_dir = Path(temporary_directory) / "predict"
            base_dir.mkdir()

            output_dir = increment_output_dir(base_dir)

        self.assertEqual(output_dir.name, "predict2")


if __name__ == "__main__":
    unittest.main()
