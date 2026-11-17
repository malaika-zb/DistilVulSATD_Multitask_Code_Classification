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
| TinyBERT student (KD, 3 epochs) | 0.25 / 0.25 | 0.7690 | ~44 | 8.15e4 | 0.8471 |
| TinyBERT student (KD, 5 epochs) | 0.5 / 0.5 | 0.7822 | 41.9 | 8.15e4 | 0.8555 |
| **TinyBERT student (submitted)** | **0.20 / 0.55** | **0.7844** | **~42** | **8.15e4** | **0.8566** |

Sources: `results/baseline_eval.json`, `results/student_tinybert_from_official_L256_T4_e5_full_score.json`, `results/final_thresholds.json`.

## Per-class metrics (submitted, thr 0.20 / 0.55)

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Vulnerability | 0.8028 | 0.8409 | 0.8214 |
| MAT | 0.9455 | 0.6178 | 0.7473 |
| **Average** | | | **0.7844** |

## Training setup (submitted student)

| Hyperparameter | Value |
|---|---|
| Teacher | `NLBSE/nlbse27-code-comment-classification-baseline` |
| Student | `huawei-noah/TinyBERT_General_4L_312D` |
| max_length | 256 |
| epochs | **5** |
| batch size | 32 |
| learning rate | 3e-5 |
| KD temperature | 4.0 |
| KD α (soft) | 0.7 |
| seed | 42 |
| Accepted thresholds | vuln=**0.20**, MAT=**0.55** |

Checkpoint (not in git): Drive folder in README →  
`checkpoints/student_runs/tinybert_from_official_L256_T4_e5/best_model/`

## Ablation note

The 3-epoch run (`..._e3`, score 0.8471) is kept on Drive and as legacy notebooks for comparison. The submitted system is the **5-epoch** checkpoint with tuned thresholds.
