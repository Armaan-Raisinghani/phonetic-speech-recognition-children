# Child Speech IPA Recognition (CNN + CTC + Transfer)

This project implements phoneme-sequence recognition for child speech using:
- log-Mel features
- CNN14-like acoustic model with transfer initialization
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

```bash
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
- Transfer loading is best-effort via torch hub (PANNs repo); if unavailable offline, training still runs with random initialization.
