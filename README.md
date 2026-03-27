# PAIChecker

Artifact repository for the paper **"PAIChecker: Uncovering and Checking PR-Issue Misalignment in SWE-Bench-Like Benchmarks"**.

This repository contains the annotated SWE-bench Verified and SWE-gym data, source code, experiment outputs, and evaluation scripts needed to reproduce all results reported in the paper.

## Repository Structure

```
PAIChecker/
├── src/paichecker/          # Source code (multi-agent pipeline, baselines, configs)
│   ├── agents/              # Default and multi-agent implementations
│   ├── config/              # YAML pipeline configurations
│   ├── environments/        # Docker and local execution environments
│   ├── models/              # LLM backend (LiteLLM)
│   ├── run/                 # Experiment runners (baselines, ablation, full pipeline)
│   └── utils/               # Logging and pricing utilities
├── scripts/                 # Evaluation and visualization scripts
│   ├── evaluate.py          # Main evaluation (Tables 3 & 4)
│   ├── evaluate_ablation.py # Ablation study evaluation (Table 5)
│   ├── visualize_pipeline_flow.py  # Pipeline stage analysis (Figure 4, Table 6)
│   └── construct_dataset.py # Build input dataset from GitHub API (requires GITHUB_TOKEN)
├── data/
│   ├── minisweagent_sample.jsonl   # Input dataset (2,438 SWE-gym instances)
│   ├── swe_gym_all.csv             # Ground truth human labels
│   ├── swe-bench-analysis.csv      # SWE-bench Verified study data (500 instances)
│   │                                 with per-instance resolution outcomes from 131
│   │                                 leaderboard agents, supporting the preliminary
│   │                                 study (Section 2)
│   ├── outputs/                    # Experiment outputs (5 methods × 4 models)
│   │   └── ablation/              # Ablation study outputs (Gemini backbone only)
│   ├── sub_agent_outputs/          # Per-instance sub-agent text outputs
│   └── results/                    # Pre-computed evaluation results
├── requirements.txt
└── LICENSE
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproducing Evaluation Results

```bash
cd scripts

# Tables 3 & 4: Main evaluation (all methods × all models)
python evaluate.py

# Table 5: Ablation study (Gemini backbone)
python evaluate_ablation.py

# Figure 4 & Table 6: Pipeline stage analysis
python visualize_pipeline_flow.py
```

Results are saved to `data/results/`.

## Re-running Experiments

Requires API credentials:

```bash
export GITHUB_TOKEN="your_github_token"
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_api_base_url"

# Run all experiments (4 models × 5 methods)
bash src/paichecker/run/run_all_experiments.sh

# Or run a specific baseline
python -m paichecker.run.baselines \
    --input data/minisweagent_sample.jsonl \
    --method cot \
    --model gemini

# Or run the full multi-agent pipeline for one instance
python -m paichecker.run.multi_swe_detector \
    --input data/minisweagent_sample.jsonl \
    --index 0 \
    --model openai/gemini-3.1-pro-preview

# Rebuild the input dataset from GitHub (requires GITHUB_TOKEN)
python scripts/construct_dataset.py
```

## Backbone Models

| Short Name | Full Model Name |
|---|---|
| gpt | openai/gpt-5.3-codex |
| claude | openai/claude-sonnet-4-6 |
| gemini | openai/gemini-3.1-pro-preview |
| qwen | openai/qwen3.5-plus |

## License

MIT
