from ultralytics import YOLO
from pathlib import Path

def main():
    print("🚀 ファインチューニング（再学習）を開始します...")

    # 1. ベースとなる「賢い脳みそ」を読み込む
    model = YOLO('yolov8n.pt')

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent / "my_bus_dataset_robust"

    # 2. オリジナルデータセットを使って学習をスタート！
    results = model.train(
        data=str(data_path / "dataset.yaml"),  # 先ほど自動生成した指示書のパス
        epochs=50,                           # 学習を繰り返す回数（今回は少なめなので50回）
        imgsz=640,                           # 画像のサイズ（標準の640）
        device='mps',                        # M2 Macの専用コア（MPS）をフル活用！
        plots=True                           # 学習の成果をグラフ化して保存する
    )

    print("\n✅ 学習が完了しました！")
    print("🧠 進化したAIモデルは 'runs/detect/train/weights/best.pt' に保存されています。")

if __name__ == "__main__":
    main()