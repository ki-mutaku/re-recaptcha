# Singularity / Slurm 実行メモ

このプロジェクトでは、SIF には Python・PyTorch・Ultralytics などの実行環境だけを入れます。
コード、画像、学習結果はジョブ実行時にプロジェクトディレクトリを `/workspace` に bind して使います。

## 1. SIF を作る

学科サーバーで `singularity` が使える場合:

```bash
singularity build --fakeroot re-recaptcha.sif containers/re-recaptcha.def
```

`apptainer` 名で入っている場合:

```bash
apptainer build --fakeroot re-recaptcha.sif containers/re-recaptcha.def
```

`--fakeroot` が使えないサーバーでは、管理者に有効化してもらうか、sudo が使える Linux マシンでビルドします。
Mac では直接 SIF を安定して作れないので、Linux 環境またはサーバー上で作るのが安全です。

## 2. 動作確認する

GPU が見えているか確認:

```bash
singularity exec --nv re-recaptcha.sif python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

分類スクリプトを手で実行:

```bash
singularity exec --nv --cleanenv \
  --bind "$PWD:/workspace" \
  --pwd /workspace \
  re-recaptcha.sif \
  python src/image_classification/classification.py --image-dir img --target bus --threshold 0.05
```

CPU だけで実行する場合は `--nv` を外します。

## 3. Slurm に投げる

分類:

```bash
sbatch slurm/classification.sbatch
```

YOLOv8 segmentation の学習:

```bash
sbatch slurm/train_yolov8_segmentation.sbatch
```

ジョブ状況:

```bash
squeue -u "$USER"
```

ログ:

```bash
ls recaptcha-*.out
tail -f recaptcha-cls-<jobid>.out
```

## 4. サーバーに合わせて変える場所

- `#SBATCH --partition=gpu`: 学科サーバーの GPU partition 名に変更する。
- `#SBATCH --gres=gpu:1`: GPU 数を変える。CPU ジョブなら削る。
- `#SBATCH --mem=...`: データ量とバッチサイズに合わせて増減する。
- `--batch 16`: GPU メモリ不足なら `8`, `4`, `2` に下げる。
- `apptainer` が PATH にあればそれを使い、無ければ `singularity` に落ちるので、`module` は必須ではない。

## 5. 注意点

- SIF は基本的に読み取り専用なので、出力先は bind した `/workspace/runs` などにする。
- `src/image_classification/classification.py` が使う ResNet18 の重みは SIF ビルド時に取得済みなので、計算ノードにインターネットがなくても動く。
- NVIDIA ドライバは SIF に入れない。実行時に `--nv` を付けることでホスト側ドライバを使う。
- CUDA の互換性が合わない場合は、`containers/re-recaptcha.def` の `From:` を学科サーバーのドライバに合う PyTorch CUDA イメージへ変更する。
