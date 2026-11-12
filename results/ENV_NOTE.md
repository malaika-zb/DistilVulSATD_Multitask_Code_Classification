# Environment note

Full training and official-style evaluation (runtime + GFLOPs) were run on **Google Colab T4**.

Local CPU is fine for data analysis (`scripts/analyze_data.py`) and smoke evals (`--quick`). For the submission score numbers in `RESULTS.md`, use a CUDA GPU and the non-`--quick` eval path.
