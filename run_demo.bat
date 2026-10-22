@echo off
REM One-command smoke test for Windows.
cd /d %~dp0
python main.py demo-data --out data/demo || exit /b 1
python main.py prepare --data-dir data/demo --out features/demo.npz || exit /b 1
python main.py train --features features/demo.npz --model mlp --epochs 30 --out-dir outputs/demo || exit /b 1
python main.py train --features features/demo.npz --model cnn --epochs 30 --out-dir outputs/demo || exit /b 1
python main.py train --features features/demo.npz --model crnn --epochs 25 --out-dir outputs/demo || exit /b 1
python main.py evaluate --checkpoint outputs/demo/mlp_best.pt || exit /b 1
python main.py evaluate --checkpoint outputs/demo/cnn_best.pt || exit /b 1
python main.py evaluate --checkpoint outputs/demo/crnn_best.pt || exit /b 1
python main.py compare --checkpoints outputs/demo/mlp_best.pt outputs/demo/cnn_best.pt outputs/demo/crnn_best.pt || exit /b 1
python main.py predict --checkpoint outputs/demo/cnn_best.pt data/demo/metal/metal.00000.wav data/demo/jazz/jazz.00003.wav

