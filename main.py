#!/usr/bin/env python3
"""Music Genre Classification using Deep Neural Networks — command-line interface.

Commands
  demo-data   Generate a small synthetic dataset (no download needed)
  prepare     Extract MFCC / mel features from data/<genre>/*.wav
  train       Train an MLP, CNN or CRNN
  evaluate    Evaluate a checkpoint on its held-out test tracks
  predict     Predict the genre of an audio file
  pipeline    prepare -> train -> evaluate in one go
"""
import argparse
import sys

from src.config import AudioConfig, TrainConfig


def add_audio_args(p):
    p.add_argument("--feature", choices=["mfcc", "mel"], default="mfcc")
    p.add_argument("--n-mfcc", type=int, default=40)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--segment-seconds", type=float, default=3.0)


def add_train_args(p):
    p.add_argument("--model", choices=["mlp", "cnn", "crnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs")


def audio_cfg(a):
    return AudioConfig(feature=a.feature, n_mfcc=a.n_mfcc, n_mels=a.n_mels, segment_seconds=a.segment_seconds)


def train_cfg(a):
    return TrainConfig(model=a.model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                       patience=a.patience, seed=a.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo-data", help="generate a synthetic dataset")
    p.add_argument("--out", default="data/demo")
    p.add_argument("--tracks-per-genre", type=int, default=12)
    p.add_argument("--seconds", type=int, default=15)

    p = sub.add_parser("prepare", help="extract features")
    p.add_argument("--data-dir", required=True, help="folder containing one sub-folder per genre")
    p.add_argument("--out", default="features/features.npz")
    add_audio_args(p)

    p = sub.add_parser("train", help="train a model")
    p.add_argument("--features", default="features/features.npz")
    add_train_args(p)

    p = sub.add_parser("evaluate", help="evaluate a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", default=None)

    p = sub.add_parser("predict", help="predict genre of audio file(s)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("audio", nargs="+")
    p.add_argument("--top-k", type=int, default=3)

    p = sub.add_parser("pipeline", help="prepare + train + evaluate")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", default="features/features.npz")
    add_audio_args(p)
    add_train_args(p)

    p = sub.add_parser("compare", help="compare multiple model checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint paths (.pt)")
    p.add_argument("--features", default=None, help="features file if different from checkpoint metadata")

    a = ap.parse_args(argv)

    if a.cmd == "demo-data":
        from src.demo_data import make_demo_dataset
        make_demo_dataset(a.out, a.tracks_per_genre, a.seconds)
    elif a.cmd == "prepare":
        from src.dataset import prepare_dataset
        prepare_dataset(a.data_dir, a.out, audio_cfg(a))
    elif a.cmd == "train":
        from src.train import train
        train(a.features, a.out_dir, train_cfg(a))
    elif a.cmd == "evaluate":
        from src.evaluate import evaluate
        evaluate(a.checkpoint, a.features)
    elif a.cmd == "predict":
        from src.predict import predict_file
        for f in a.audio:
            predict_file(a.checkpoint, f, a.top_k)
    elif a.cmd == "pipeline":
        from src.dataset import prepare_dataset
        from src.train import train
        from src.evaluate import evaluate
        prepare_dataset(a.data_dir, a.features, audio_cfg(a))
        ckpt = train(a.features, a.out_dir, train_cfg(a))
        evaluate(ckpt, a.features)
    elif a.cmd == "compare":
        from src.evaluate import compare_models
        compare_models(a.checkpoints, a.features)


if __name__ == "__main__":
    sys.exit(main())
