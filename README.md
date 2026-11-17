# DistilVulSATD

Knowledge-distilled TinyBERT for **NLBSE’27 Multitask Code Classification** (vulnerability + SATD/MAT).

**Public-test score (submitted config): 0.8566**  
(avg F1 0.784 · runtime ~42 s · GFLOPs ~8.2e4) on Colab T4 with thresholds `(0.20, 0.55)`.

Compared to the official CodeBERT baseline (~0.72 on the same formula), DistilVulSATD trades some F1 for a large drop in runtime and FLOPs, which dominates the ranking score.

## Artifacts (model weights)

Weights are **not** stored in this repository (size / git limits). Download the student checkpoint from Google Drive:

**[NLBSE27_DistilVulSATD (Drive)](https://drive.google.com/drive/folders/1Tpe54KkA_N-t6qlt0_C3h-WHuF5T-NQr)**

**Submitted checkpoint:**

```text
.../checkpoints/student_runs/tinybert_from_official_L256_T4_e5/best_model/
  config.json
  model.safetensors
  tokenizer files
```

Accepted thresholds: **vuln = 0.20**, **MAT = 0.55**  
(see `results/final_thresholds.json`).

Metric tables live under [`results/`](results/) — see [`RESULTS.md`](RESULTS.md).

## Method (short)

1. Reproduce the official CodeBERT baseline eval harness (F1 + timed AMP forwards + FLOPs + score).
2. Distill the official baseline teacher into `TinyBERT_General_4L_312D` (**5 epochs**, T=4, α=0.7, max_length=256).
3. Tune decision thresholds on a held-out validation carve; accept `(0.20, 0.55)` → public score **0.8566**.

## Which notebooks to run (submitted pipeline)

| Notebook | Role |
|---|---|
| `notebooks/00_baseline_reproduce.ipynb` | Baseline score on T4 |
| `notebooks/02_train_teacher.ipynb` | Optional own CodeBERT teacher (ablation) |
| **`notebooks/03_distill_student.ipynb`** | **Submitted KD (5 epochs) + public score @0.5** |
| **`notebooks/06_threshold_tune.ipynb`** | **Submitted threshold tune → 0.20 / 0.55** |

Legacy / ablation only (3-epoch run, score 0.8471):

- `notebooks/03_distill_student_e3_legacy.ipynb`
- `notebooks/06_threshold_tune_e3_legacy.ipynb`

Drive root used in notebooks: `MyDrive/NLBSE27_DistilVulSATD/`.

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
# After downloading e5 best_model/ from Drive:
python scripts/eval_student.py \
  --model-dir /path/to/tinybert_from_official_L256_T4_e5/best_model \
  --thr-vuln 0.20 --thr-mat 0.55 \
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
