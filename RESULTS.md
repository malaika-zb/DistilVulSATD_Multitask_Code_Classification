# DistilVulSATD — public-test results

Hardware for reported runs: **Google Colab T4**. Dataset: public test split of `NLBSE/nlbse27-code-comment-classification` (30 655 rows).

## Ranking formula

```text
score = 0.6 · avgF1
      + 0.2 · max((1000 − runtime) / 1000, 0)
      + 0.2 · max((3e6 − GFLOPs) / 3e6, 0)
```

## Summary table

| Model | Thresholds (vuln / MAT) | avg F1 | runtime (s) | GFLOPs | **score** |
|---|---|---:|---:|---:|---:|
| Official CodeBERT baseline (our T4) | 0.5 / 0.5 | 0.9102 | 259.2 | 2.67e6 | 0.7165 |
| Organiser-reported baseline (ref.) | 0.5 / 0.5 | 0.910 | ~192 | ~2.67e6 | ~0.730 |
| TinyBERT student (KD, 3 epochs) | 0.5 / 0.5 | 0.7624 | 44.0 | 8.15e4 | 0.8432 |
| **TinyBERT student (accepted)** | **0.25 / 0.25** | **0.7690** | **44.0** | **8.15e4** | **0.8471** |

Sources: `results/baseline_eval.json`, `results/threshold_tune_result.json`, `results/final_thresholds.json`.

## Per-class metrics (accepted, thr 0.25 / 0.25)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Vulnerability | 0.8238 | 0.8016 | 0.8126 |
| MAT | 0.9077 | 0.6040 | 0.7253 |
| **Average** | | | **0.7690** |

## Training setup (student)

| Hyperparameter | Value |
|---|---|
| Teacher | `NLBSE/nlbse27-code-comment-classification-baseline` |
| Student | `huawei-noah/TinyBERT_General_4L_312D` |
| max_length | 256 |
| epochs | 3 |
| batch size | 32 |
| learning rate | 3e-5 |
| KD temperature | 4.0 |
| KD α (soft) | 0.7 |
| seed | 42 |
| Accepted thresholds | vuln=0.25, MAT=0.25 |

Checkpoint (not in git): Drive folder in README →  
`checkpoints/student_runs/tinybert_from_official_L256_T4_e3/best_model/`
