# Child Speech IPA Recognition (CNN + CTC + Transfer)

This project implements phoneme-sequence recognition for child speech using:
- log-Mel features
- CNN14-like acoustic model with PANNs transfer initialization
- last-layer-only tuning for the supervised transfer baseline
- CTC loss (no frame alignment required)
- semi-supervised pseudo-labeling (top-confidence selection)

## 1) Setup

```bash
pip install -r requirements.txt
```

## 2) Preprocess and split (speaker-disjoint)

```bash
python scripts/preprocess.py \
  --root . \
  --labels train_phon_transcripts.jsonl \
  --audio-dir audio \
  --out-dir data/manifests \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

Outputs:
- `data/manifests/train.jsonl`
- `data/manifests/val.jsonl`
- `data/manifests/test.jsonl`
- `data/manifests/unlabeled.jsonl`
- `data/manifests/vocab.json`
- `data/manifests/report.json`

## 3) Supervised baseline training

`configs/train.yaml` starts with labeled data and runs the transfer-learning experiment requested in review: four AudioSet-pretrained CNN14 blocks are frozen, the head is a direct linear CTC classifier, and only `classifier.weight` / `classifier.bias` are optimized.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py --config configs/train.yaml
```

Best checkpoint:
- `checkpoints/baseline/best.pt`

## 4) Evaluate baseline

```bash
python scripts/evaluate.py \
  --manifest data/manifests/test.jsonl \
  --vocab data/manifests/vocab.json \
  --checkpoint checkpoints/baseline/best.pt \
  --out results/baseline_test_predictions.jsonl
```

This also writes `results/baseline_test_predictions.phoneme_report.json`, which includes worst phoneme recall, substitutions, deletions, insertions, and per-phoneme stats.

## 5) Semi-supervised round (Option A)

Generate pseudo-labels from unlabeled pool using top 20% confidence:

```bash
python scripts/pseudo_label.py \
  --unlabeled-manifest data/manifests/unlabeled.jsonl \
  --supervised-train-manifest data/manifests/train.jsonl \
  --vocab data/manifests/vocab.json \
  --checkpoint checkpoints/baseline/best.pt \
  --keep-ratio 0.2 \
  --pseudo-weight 0.4 \
  --out-pseudo data/manifests/pseudo_labels.jsonl \
  --out-merged-train data/manifests/train_with_pseudo.jsonl
```

Then train again:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train.py --config configs/train_with_pseudo.yaml
```

Evaluate semi-supervised model:

```bash
python scripts/evaluate.py \
  --manifest data/manifests/test.jsonl \
  --vocab data/manifests/vocab.json \
  --checkpoint checkpoints/semi_supervised_round1/best.pt \
  --out results/semi_test_predictions.jsonl
```

Compare phoneme-level behavior across evaluated models:

```bash
python scripts/compare_phoneme_reports.py \
  results/baseline_test_predictions.phoneme_report.json \
  results/semi_test_predictions.phoneme_report.json
```

## 6) Single-file inference (phoneme sequence only)

```bash
python scripts/infer.py \
  --audio audio/U_xxx.flac \
  --vocab data/manifests/vocab.json \
  --checkpoint checkpoints/baseline/best.pt
```

## Notes

- Primary metric is PER (implemented as edit distance over non-space IPA symbols).
- Secondary metric is CER.
- Transfer loading is best-effort from the PANNs CNN14 16 kHz checkpoint; if unavailable offline, training still runs with random initialization.
- Use `model.trainable_scope: classifier` for last-layer-only tuning, `head` for frozen-CNN plus trainable MLP head, or `all` for full fine-tuning.
- The training configs are tuned for 8 GB GPUs by using smaller per-step batches, gradient accumulation, and length-bucketed batches to reduce padding-related memory spikes.
- Set `train.unfreeze_epoch: null` to keep the backbone frozen for the full run. Unfreezing increases memory use substantially.

## Experiment docs

- `METHOD_FINDINGS.md`: metrics, failure analysis, and review narrative.
- `TRACK_D_AND_VGG_TUNING.md`: how to run/tune PANNs CNN + BiLSTM + last-block fine-tuning, and how to run/evaluate VGG for more epochs.
- `scripts/train_panns_cnn_bilstm_last_block.py`: standalone Track D trainer with CLI-tunable parameters and per-epoch phoneme reports.
- `scripts/evaluate_vgg_bilstm.py`: evaluates `best_vgg.pt` and writes VGG phoneme-level comparison reports.
