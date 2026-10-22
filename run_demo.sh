#!/usr/bin/env bash
# One-command smoke test: synthetic data -> features -> train MLP & CNN -> evaluate -> predict.
set -e
cd "$(dirname "$0")"
python main.py demo-data --out data/demo
python main.py prepare --data-dir data/demo --out features/demo.npz
python main.py train --features features/demo.npz --model mlp --epochs 30 --out-dir outputs/demo
python main.py train --features features/demo.npz --model cnn --epochs 30 --out-dir outputs/demo
python main.py train --features features/demo.npz --model crnn --epochs 25 --out-dir outputs/demo
python main.py evaluate --checkpoint outputs/demo/mlp_best.pt
python main.py evaluate --checkpoint outputs/demo/cnn_best.pt
python main.py evaluate --checkpoint outputs/demo/crnn_best.pt
python main.py compare --checkpoints outputs/demo/mlp_best.pt outputs/demo/cnn_best.pt outputs/demo/crnn_best.pt
python main.py predict --checkpoint outputs/demo/cnn_best.pt data/demo/metal/metal.00000.wav data/demo/jazz/jazz.00003.wav

