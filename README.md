# Limitations of Cross-Lingual Information Extraction in Multilingual LLMs

Code and diagnostic dataset for **CSE4057 (Natural Language Processing) HW2**, Hanyang University.

## 1. Environment Setup
- **Hardware**: MacBook Air (Apple Silicon M-series, 8GB Unified Memory)
- **OS & Python**: macOS, Python 3.12
- **Model**: `mlx-community/Qwen2.5-3B-Instruct-4bit` (Greedy decoding, `temperature=0.0`, `max_tokens=128`)

Install dependencies on a fresh machine:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Quick Reproduction (Tables 1, 2 & Failure Cases)
To reproduce **Table 1**, **Table 2**, and the **Sampled Failure Cases** directly from the saved model outputs (`experiment_results.json`):

```bash
python3 3_evaluate.py
```

## 3. Full Pipeline Reproduction (From Scratch)
To rebuild the dataset from `klue/klue` and re-run all 5 experimental conditions locally on Apple Silicon:

```bash
python3 1_build_dataset.py      # Generates raw_100_ko.json
python3 2_run_experiments.py    # Generates benchmark_100_clean.json & experiment_results.json
python3 3_evaluate.py           # Computes metrics and 95% Bootstrap CIs
```

## 4. Random Seeds & Determinism
- **LLM Decoding**: Deterministic greedy decoding (`temperature=0.0`).
- **Bootstrap Resampling (Table 2)**: `seed=42` (`n_boot=1000` in `numpy.random.default_rng(42)`).
- **Failure Case Sampling (Table 3)**: `random.seed(42)`.
