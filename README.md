# Music Genre Classification Using Deep Neural Networks

An end-to-end, executable PyTorch project that classifies music into genres from raw audio.
It extracts **MFCC** or **log-mel spectrogram** features with `librosa`, trains one of three deep
networks (**MLP**, **CNN**, **CRNN**), evaluates it on held-out songs, and predicts the genre of new audio files.

## Pipeline

```
audio (.wav/.mp3/.au)  ->  split into 3 s segments  ->  MFCC (130 x 40) / log-mel (130 x 64)
      ->  normalise (train-set stats)  ->  MLP | CNN | CRNN  ->  softmax per segment
      ->  average over segments  ->  genre of the whole track
```

Key design choices:
- **Segmenting**: each 30 s GTZAN track becomes 10 x 3 s training examples (10x more data).
- **Split by track, not by segment**: segments from one song never appear in both train and test, which avoids the inflated accuracy common in tutorials.
- **Two accuracy figures**: segment-level and track-level (segment probabilities averaged per song).
- Early stopping, ReduceLROnPlateau, AdamW, label smoothing, BatchNorm and Dropout.

## Models

| Model | Idea | Params (MFCC input) |
|-------|------|------|
| `mlp`  | Flattened features -> 512 -> 256 -> 64 -> classes | ~2.8 M |
| `cnn`  | 3 double-conv blocks (32/64/128) + global average pooling | ~0.3 M |
| `crnn` | 2 conv blocks + bidirectional GRU over time | ~0.66 M |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.9+; uses a GPU (CUDA) or Apple Silicon (MPS) automatically if available, otherwise CPU.

## Quick start (no download needed)

```bash
./run_demo.sh        # Windows: run_demo.bat
```
This generates a small **synthetic** 10-genre dataset, extracts features, trains the MLP and CNN,
evaluates both and runs predictions — about 5 minutes on a laptop CPU. On this synthetic data the CNN
reaches ~99 % and the MLP ~70 % test accuracy. The synthetic data only proves the pipeline works;
use GTZAN for real results.

## Training on the real GTZAN dataset

1. Download GTZAN (1,000 tracks, 10 genres, 30 s each), e.g. from Kaggle:
   <https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification>
2. Put the `genres_original` folder at `data/genres_original/` so it looks like
   `data/genres_original/blues/blues.00000.wav`, … (any folder with one sub-folder per genre works — including your own music).
3. Run:

```bash
# all steps at once
python main.py pipeline --data-dir data/genres_original --model cnn --out-dir outputs/gtzan

# or step by step
python main.py prepare  --data-dir data/genres_original --out features/gtzan.npz
python main.py train    --features features/gtzan.npz --model cnn  --epochs 60 --out-dir outputs/gtzan
python main.py train    --features features/gtzan.npz --model mlp  --epochs 60 --out-dir outputs/gtzan
python main.py evaluate --checkpoint outputs/gtzan/cnn_best.pt
python main.py predict  --checkpoint outputs/gtzan/cnn_best.pt path/to/song.mp3
```

Use mel spectrograms (often better for CNN/CRNN):
```bash
python main.py prepare --data-dir data/genres_original --out features/gtzan_mel.npz --feature mel
python main.py train   --features features/gtzan_mel.npz --model crnn --out-dir outputs/gtzan_mel
```

The known-corrupt file `jazz.00054.wav` is skipped automatically. Feature extraction for GTZAN takes a few
minutes; CNN training takes a few minutes on a GPU and roughly 30–60 minutes on CPU.
Typical results on GTZAN with a track-level split are ~55–65 % (MLP) and ~70–80 % (CNN/CRNN) per segment, a little higher per track.

## Outputs (in `--out-dir`)

| File | Contents |
|------|----------|
| `<model>_best.pt` | best checkpoint (weights, genres, normalisation stats, configs, test-track ids) |
| `<model>_training_curves.png` | loss / accuracy per epoch |
| `<model>_history.json` | raw training history |
| `<model>_confusion_matrix.png` | test confusion matrix |
| `<model>_report.txt` | accuracy + per-genre precision / recall / F1 |

## Interactive Web App

Launch the Streamlit web dashboard to upload or select songs, listen to audio, visualize waveforms and mel-spectrograms, and run interactive predictions:

```bash
streamlit run app.py     # or: run_app.bat
```

## Project structure

```
music_genre_classifier/
├── app.py             # Interactive Streamlit Web GUI
├── main.py            # CLI: demo-data | prepare | train | evaluate | compare | predict | pipeline
├── run_demo.sh / .bat # one-command end-to-end demo execution
├── run_app.bat        # one-click web dashboard launcher
├── requirements.txt
└── src/
    ├── config.py      # audio + training hyper-parameters
    ├── features.py    # loading, segmenting, MFCC / mel extraction
    ├── dataset.py     # folder scan, feature caching, track-level split, torch Dataset
    ├── models.py      # MLP, CNN, CRNN
    ├── train.py       # training loop, early stopping, checkpointing
    ├── evaluate.py    # segment + track accuracy, compare_models, report, confusion matrix
    ├── predict.py     # single-file inference
    ├── plots.py       # charts
    └── demo_data.py   # synthetic dataset generator
```

## CLI options

`python main.py <command> -h` lists every option. Common ones:
- Commands: `demo-data`, `prepare`, `train`, `evaluate`, `compare`, `predict`, `pipeline`
- Model selection: `--model {mlp,cnn,crnn}`
- Training hyperparameters: `--epochs`, `--batch-size`, `--lr`, `--patience`, `--seed`, `--out-dir`
- Audio features: `--feature {mfcc,mel}`, `--n-mfcc`, `--n-mels`, `--segment-seconds`

