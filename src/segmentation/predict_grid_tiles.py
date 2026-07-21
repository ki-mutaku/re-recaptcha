"""YOLOv8-segの予測マスクから4×4グリッドの選択タイルを求める。

ファインチューニング済みモデルでtest splitを推論し、対象クラスのマスクが
重なるタイル番号を出力する。YOLOポリゴンラベルから正解タイルも生成し、
タイル単位の評価指標と確認用ハイライト画像を保存する。
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw

GRID_ROWS = 4
GRID_COLUMNS = 4
TILE_COUNT = GRID_ROWS * GRID_COLUMNS
DEFAULT_MODEL_PATH = Path(
    "runs/segment/yolov8_segmentation_runs/"
    "fog_mosaic_finetune-4/weights/best.pt"
)
DEFAULT_SOURCE = Path("src/yolo_dataset/images/test")
DEFAULT_LABELS_DIR = Path("src/yolo_dataset/labels/test")
DEFAULT_OUTPUT_ROOT = Path("runs/segment/grid_tile_predictions/predict")


def tile_bounds(
    width: int,
    height: int,
    rows: int = GRID_ROWS,
    columns: int = GRID_COLUMNS,
) -> list[tuple[int, int, int, int, int]]:
    """端数画素を失わず、画像全体を行優先・1始まりのタイル境界へ分割する。"""
    if width < columns or height < rows:
        raise ValueError("画像サイズはグリッドの行数・列数以上である必要があります。")

    bounds = []
    for row in range(rows):
        y_min = row * height // rows
        y_max = (row + 1) * height // rows
        for column in range(columns):
            x_min = column * width // columns
            x_max = (column + 1) * width // columns
            tile_number = row * columns + column + 1
            bounds.append((tile_number, x_min, y_min, x_max, y_max))
    return bounds


def select_tiles_from_mask(
    mask: torch.Tensor,
    min_mask_ratio: float = 0.0,
    rows: int = GRID_ROWS,
    columns: int = GRID_COLUMNS,
) -> tuple[list[int], dict[int, float]]:
    """対象マスクの画素が存在するタイルと、各タイル内のマスク面積率を返す。"""
    if mask.ndim != 2:
        raise ValueError("maskは高さ×幅の2次元Tensorである必要があります。")
    if not 0.0 <= min_mask_ratio <= 1.0:
        raise ValueError("min_mask_ratioは0から1の範囲で指定してください。")

    boolean_mask = mask.bool()
    height, width = boolean_mask.shape
    selected_tiles = []
    coverages: dict[int, float] = {}
    for tile_number, x_min, y_min, x_max, y_max in tile_bounds(
        width,
        height,
        rows,
        columns,
    ):
        tile_mask = boolean_mask[y_min:y_max, x_min:x_max]
        mask_pixels = int(tile_mask.sum().item())
        ratio = mask_pixels / tile_mask.numel()
        coverages[tile_number] = ratio
        if mask_pixels > 0 and ratio >= min_mask_ratio:
            selected_tiles.append(tile_number)

    return selected_tiles, coverages


def extract_target_mask(result: Any, target_class_id: int) -> torch.Tensor:
    """1画像分のYOLO結果から対象クラスの全インスタンスを1枚のマスクへ統合する。"""
    height, width = result.orig_shape
    empty_mask = torch.zeros((height, width), dtype=torch.bool)
    if result.masks is None or result.boxes is None:
        return empty_mask

    mask_data = result.masks.data.detach().cpu()
    class_ids = result.boxes.cls.detach().cpu().to(torch.int64)
    if len(mask_data) != len(class_ids):
        raise ValueError("予測マスク数とクラスID数が一致していません。")

    target_masks = mask_data[class_ids == target_class_id]
    if len(target_masks) == 0:
        return empty_mask

    union_mask = torch.any(target_masks > 0.5, dim=0)
    if union_mask.shape != (height, width):
        union_mask = functional.interpolate(
            union_mask[None, None].float(),
            size=(height, width),
            mode="nearest",
        )[0, 0].bool()
    return union_mask


def load_yolo_target_mask(
    label_path: Path,
    width: int,
    height: int,
    target_class_id: int,
) -> torch.Tensor:
    """YOLOの正規化ポリゴンラベルから対象クラスの正解マスクを復元する。"""
    if not label_path.is_file():
        raise FileNotFoundError(f"正解ラベルが見つかりません: {label_path}")

    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        values = line.split()
        class_id = int(values[0])
        coordinates = [float(value) for value in values[1:]]
        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            raise ValueError(
                f"不正なYOLOポリゴンです: {label_path}:{line_number}"
            )
        if class_id != target_class_id:
            continue

        points = [
            (
                min(
                    width - 1,
                    max(0.0, min(1.0, coordinates[index])) * width,
                ),
                min(
                    height - 1,
                    max(0.0, min(1.0, coordinates[index + 1])) * height,
                ),
            )
            for index in range(0, len(coordinates), 2)
        ]
        draw.polygon(points, fill=1)

    mask_buffer = bytearray(mask_image.tobytes())
    return torch.frombuffer(mask_buffer, dtype=torch.uint8).reshape(height, width).bool()


def calculate_tile_metrics(
    predicted_tiles: Sequence[set[int]],
    expected_tiles: Sequence[set[int]],
    tile_count: int = TILE_COUNT,
) -> dict[str, float | int]:
    """複数画像のタイル選択から混同行列、分類指標、完全一致率を計算する。"""
    if len(predicted_tiles) != len(expected_tiles):
        raise ValueError("予測結果と正解結果の画像数が一致していません。")
    if not predicted_tiles:
        raise ValueError("評価対象の画像がありません。")

    valid_tiles = set(range(1, tile_count + 1))
    tp = fp = fn = tn = exact_matches = 0
    for predicted, expected in zip(predicted_tiles, expected_tiles, strict=True):
        if not predicted <= valid_tiles or not expected <= valid_tiles:
            raise ValueError("タイル番号が有効範囲外です。")
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        tn += len(valid_tiles - (predicted | expected))
        exact_matches += predicted == expected

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "images": len(predicted_tiles),
        "tiles": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_matches / len(predicted_tiles),
    }


def compare_prediction_with_expected(
    predicted_mask: torch.Tensor,
    expected_mask: torch.Tensor,
    min_prediction_mask_ratio: float,
) -> tuple[list[int], list[int], dict[int, float]]:
    """予測だけに面積率閾値を適用し、正解は1画素以上でタイル化する。"""
    predicted_tiles, coverages = select_tiles_from_mask(
        predicted_mask,
        min_mask_ratio=min_prediction_mask_ratio,
    )
    expected_tiles, _ = select_tiles_from_mask(expected_mask)
    return predicted_tiles, expected_tiles, coverages


def increment_output_dir(base_dir: Path) -> Path:
    """既存の実験成果を上書きしない連番付き出力ディレクトリを決定する。"""
    if not base_dir.exists():
        return base_dir
    for index in range(2, 10_000):
        candidate = base_dir.with_name(f"{base_dir.name}{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("利用可能な出力ディレクトリ名を決定できませんでした。")


def resolve_target_class_id(
    names: Mapping[int, str] | Sequence[str],
    target: str,
) -> int:
    """英語クラス名またはクラスIDをモデル内部のクラスIDへ解決する。"""
    indexed_names = (
        {int(class_id): name for class_id, name in names.items()}
        if isinstance(names, Mapping)
        else dict(enumerate(names))
    )
    if target.isdigit():
        class_id = int(target)
        if class_id in indexed_names:
            return class_id
    else:
        normalized_target = target.strip().lower()
        for class_id, class_label in indexed_names.items():
            if class_label.lower() == normalized_target:
                return class_id

    available = ", ".join(indexed_names.values())
    raise ValueError(f"対象クラスがモデルにありません: {target}。候補: {available}")


def create_highlight_image(
    image_path: Path,
    target_mask: torch.Tensor,
    selected_tiles: Sequence[int],
    output_path: Path,
) -> None:
    """予測マスクを緑、選択タイルを赤、4×4境界と番号を白で描画して保存する。"""
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGBA")

    width, height = image.size
    if target_mask.shape != (height, width):
        raise ValueError("ハイライト対象マスクと入力画像のサイズが一致していません。")

    tile_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    tile_draw = ImageDraw.Draw(tile_overlay)
    selected_set = set(selected_tiles)
    bounds = tile_bounds(width, height)
    for tile_number, x_min, y_min, x_max, y_max in bounds:
        if tile_number in selected_set:
            tile_draw.rectangle(
                (x_min, y_min, x_max - 1, y_max - 1),
                fill=(255, 0, 0, 75),
            )
    image = Image.alpha_composite(image, tile_overlay)

    mask_bytes = bytearray(target_mask.to(torch.uint8).mul(255).numpy().tobytes())
    mask_image = Image.frombytes("L", image.size, bytes(mask_bytes))
    mask_overlay = Image.new("RGBA", image.size, (0, 255, 0, 95))
    image.alpha_composite(Image.composite(mask_overlay, Image.new("RGBA", image.size), mask_image))

    grid_draw = ImageDraw.Draw(image)
    for tile_number, x_min, y_min, x_max, y_max in bounds:
        grid_draw.rectangle(
            (x_min, y_min, x_max - 1, y_max - 1),
            outline=(255, 255, 255, 255),
            width=2,
        )
        grid_draw.text(
            (x_min + 4, y_min + 4),
            str(tile_number),
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, quality=95)


def resolve_device(device: str) -> str | None:
    """auto指定をUltralyticsの自動デバイス選択へ変換する。"""
    return None if device == "auto" else device


def parse_args() -> argparse.Namespace:
    """モデル、対象クラス、推論閾値、入出力先をCLIから受け取る。"""
    parser = argparse.ArgumentParser(
        description="YOLOv8-segで対象物を検出し、4×4の選択タイルを出力します。"
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--target", default="bus", help="YOLOの英語クラス名またはID")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--min-tile-mask-ratio",
        type=float,
        default=0.0,
        help="タイルを選択する最小マスク面積率（既定は1画素以上）",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    """test画像を推論し、タイル番号・自動評価・JSON・可視化画像を出力する。"""
    args = parse_args()
    if not args.model_path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {args.model_path}")
    if not args.source.exists():
        raise FileNotFoundError(f"入力画像が見つかりません: {args.source}")
    if not args.labels_dir.is_dir():
        raise FileNotFoundError(f"正解ラベルが見つかりません: {args.labels_dir}")

    from ultralytics import YOLO

    model = YOLO(str(args.model_path))
    if getattr(model, "task", None) != "segment":
        raise ValueError(f"セグメンテーションモデルではありません: {args.model_path}")
    target_class_id = resolve_target_class_id(model.names, args.target)

    output_dir = increment_output_dir(args.output_root)
    image_output_dir = output_dir / "images"
    output_dir.mkdir(parents=True)

    image_records = []
    predicted_sets: list[set[int]] = []
    expected_sets: list[set[int]] = []
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=resolve_device(args.device),
        retina_masks=True,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path)
        height, width = result.orig_shape
        predicted_mask = extract_target_mask(result, target_class_id)
        label_path = args.labels_dir / f"{image_path.stem}.txt"
        expected_mask = load_yolo_target_mask(
            label_path,
            width,
            height,
            target_class_id,
        )
        predicted_tiles, expected_tiles, coverages = compare_prediction_with_expected(
            predicted_mask,
            expected_mask,
            args.min_tile_mask_ratio,
        )

        output_image_path = image_output_dir / image_path.name
        create_highlight_image(
            image_path,
            predicted_mask,
            predicted_tiles,
            output_image_path,
        )
        is_exact_match = predicted_tiles == expected_tiles
        print(
            f"{image_path.name}: predicted={predicted_tiles} "
            f"expected={expected_tiles} exact={is_exact_match}"
        )

        predicted_sets.append(set(predicted_tiles))
        expected_sets.append(set(expected_tiles))
        image_records.append(
            {
                "image": str(image_path),
                "predicted_tiles": predicted_tiles,
                "expected_tiles": expected_tiles,
                "exact_match": is_exact_match,
                "tile_mask_ratios": {
                    str(tile_number): ratio
                    for tile_number, ratio in coverages.items()
                },
                "highlight_image": str(output_image_path),
            }
        )

    metrics = calculate_tile_metrics(predicted_sets, expected_sets)
    print(
        "Metrics | "
        f"Accuracy={metrics['accuracy']:.4f} "
        f"Precision={metrics['precision']:.4f} "
        f"Recall={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} "
        f"ExactMatch={metrics['exact_match']:.4f}"
    )

    payload = {
        "model": str(args.model_path),
        "source": str(args.source),
        "labels_dir": str(args.labels_dir),
        "target": args.target,
        "target_class_id": target_class_id,
        "grid": {"rows": GRID_ROWS, "columns": GRID_COLUMNS},
        "parameters": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "min_tile_mask_ratio": args.min_tile_mask_ratio,
        },
        "metrics": metrics,
        "images": image_records,
    }
    json_path = output_dir / "predictions.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"出力先: {output_dir}")


if __name__ == "__main__":
    main()
