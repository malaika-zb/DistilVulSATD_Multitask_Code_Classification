# DistilVulSATD

Knowledge-distilled TinyBERT for **NLBSE’27 Multitask Code Classification** (vulnerability + SATD/MAT).

**Public-test score: 0.8471** (avg F1 0.769 · runtime ~48.5 s · GFLOPs ~8.2e4) on Colab T4 with thresholds `(0.25, 0.25)`.

Compared to the official CodeBERT baseline (~0.72–0.73 on the same formula), DistilVulSATD trades some F1 for a large drop in runtime and FLOPs, which dominates the ranking score.

## Artifacts (model weights)

Weights are **not** stored in this repository (size / git limits). Download the student checkpoint from Google Drive:

**[NLBSE27_DistilVulSATD (Drive)](https://drive.google.com/drive/folders/1Tpe54KkA_N-t6qlt0_C3h-WHuF5T-NQr)**

Expected path after download:

```text
.../checkpoints/student_runs/tinybert_from_official_L256_T4_e3/best_model/
  config.json
  model.safetensors
  tokenizer files
```

Metric tables (no weights) live under [`results/`](results/) — see [`RESULTS.md`](RESULTS.md).

## Method (short)

1. Reproduce the official CodeBERT baseline eval harness (F1 + timed AMP forwards + FLOPs + score).
2. Distill the official baseline teacher into `TinyBERT_General_4L_312D` (3 epochs, T=4, α=0.7, max_length=256).
3. Tune decision thresholds on a held-out validation carve; accept `(0.25, 0.25)` → public score **0.8471**.

## Repository layout

```text
DistilVulSATD/
  configs/default.yaml     # hyperparams + accepted thresholds
  distilvulsatd/           # Python package (data, models, evaluate)
  scripts/                 # CLI runners
  notebooks/               # Colab-ready pipeline
  results/                 # metric JSON (no weights)
  RESULTS.md               # score table
```

## Setup

```bash
pip install -r requirements.txt
```

Data: Hugging Face `NLBSE/nlbse27-code-comment-classification`, or local `NLBSE_2027_Dataset/{train,test}.csv`.

## Reproduce public-test score

```bash
# After downloading best_model/ from Drive:
python scripts/eval_student.py \
  --model-dir /path/to/best_model \
  --thr-vuln 0.25 --thr-mat 0.25 \
  --max-length 256
```

Smoke test (CPU OK):

```bash
python scripts/eval_student.py --model-dir /path/to/best_model --quick
```

Baseline reproduce:

```bash
python scripts/run_baseline_eval.py          # full (GPU)
python scripts/run_baseline_eval.py --quick  # smoke
```

## Colab notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/00_baseline_reproduce.ipynb` | Official baseline score on T4 |
| `notebooks/03_distill_student.ipynb` | One-shot KD + full public score |
| `notebooks/06_threshold_tune.ipynb` | Val grid → accept 0.25/0.25 |

Drive persistence root used in notebooks: `MyDrive/NLBSE27_DistilVulSATD/`.

## Scoring formula

```text
score = 0.6 * avgF1
      + 0.2 * max((1000 - runtime) / 1000, 0)
      + 0.2 * max((3e6 - GFLOPs) / 3e6, 0)
```

## Citation / competition

Built for the [NLBSE’27](https://nlbse2027.github.io/) multitask code-comment classification challenge. Dataset and baseline: Hugging Face org `NLBSE`.

## License

MIT — see [`LICENSE`](LICENSE).
