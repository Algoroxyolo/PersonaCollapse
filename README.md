# Persona Collapse

Code and data for the paper "The Chameleon’s Limit: Investigating Persona Collapse and Homogenization in Large Language Models".

This codebase supports three experiments:

1. **Moral Reasoning** -- Evaluate how LLMs respond to moral dilemmas across diverse personas
2. **Digital Twin (BFI)** -- Replicate human personality assessments (Big Five Inventory) with LLM-simulated personas
3. **Self-Introduction** -- Collect open-ended self-introductions from LLM-simulated personas

## Repository Structure

```
persona-collapse/
├── persona_sim/             # Shared library
│   ├── consts.py            # Constants (rating scales)
│   ├── prompts.py           # Prompt construction for all experiments
│   ├── providers.py         # Model registries, client factory, Bedrock wrapper
│   ├── experiment.py        # Shared experiment runner (DB, retries, diagnostics)
│   └── metrics.py           # Analysis metrics (Hopkins, PRDC, uniformity)
├── experiments/             # Experiment entry points
│   ├── moral_reasoning.py   # Moral dilemma experiment
│   ├── digital_twin.py      # BFI personality test replication
│   ├── self_introduction.py # Self-introduction collection
│   └── estimate_budget.py   # Token/cost estimation
├── analysis/                        # Post-processing and analysis
│   ├── analysis_pipeline.py         # Full metrics pipeline (BFI, moral, self-intro)
│   ├── check_persona_inconsistency.py  # Generate flagged_personas.pkl
│   ├── extract_csv.py               # Export SQLite results to CSV
│   ├── README.md                    # Analysis folder documentation
│   └── visualizations/              # Figure generation scripts
├── scripts/                         # Shell scripts for running experiments
│   ├── run_moral_reasoning.sh
│   ├── run_digital_twin.sh
│   ├── run_self_introduction.sh
│   ├── run_analysis.sh
│   └── serve_vllm.sh
└── data/                            # Input data files
    ├── sampled_personas_2000.json   # 2000 synthetic persona profiles
    ├── claude-3-5-sonnet_AB_0_with_logprobs.jsonl  # Moral dilemma scenarios
    ├── flagged_personas.pkl         # Pre-computed inconsistent persona IDs
    └── human_reference/
        ├── BFI.json                 # BFI-44 question definitions
        └── wave_1_numbers.csv       # Human survey reference data
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies are `openai`, `numpy`, `pandas`, `tqdm`, and `scikit-learn`. Optional dependencies are listed in `requirements.txt` with comments.

### 2. Set API Keys

Set the appropriate environment variable for your provider:

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# OpenRouter (for models like Kimi, MiniMax, etc.)
export OPENROUTER_API_KEY="your-key-here"

# AWS Bedrock (for Anthropic Claude models)
export AWS_BEARER_TOKEN_BEDROCK="your-key-here"

# Local vLLM (no key needed, but set the base URL)
export VLLM_BASE_URL="http://localhost:8000/v1"
```

## Running Experiments

### Supported Providers

| Provider | Models | Key |
|----------|--------|-----|
| `openai` | GPT-5, GPT-4.1 family | `OPENAI_API_KEY` |
| `openrouter` | Kimi K2.5, MiniMax M2/M2.5, etc. | `OPENROUTER_API_KEY` |
| `bedrock` | Claude Sonnet/Haiku/Opus | `AWS_BEARER_TOKEN_BEDROCK` |
| `vllm` | Any HuggingFace model via vLLM | `VLLM_BASE_URL` |

### Moral Reasoning Experiment

```bash
# Step 1: Estimate budget
bash scripts/run_moral_reasoning.sh openrouter kimi-k2.5

# Step 2: Run experiment
bash scripts/run_moral_reasoning.sh openrouter kimi-k2.5 --run

# Or run Python directly:
python -m experiments.moral_reasoning \
    --provider openrouter \
    --model kimi-k2.5 \
    --personas-file data/sampled_personas_2000.json \
    --scenarios-file data/claude-3-5-sonnet_AB_0_with_logprobs.jsonl \
    --use-db 1 \
    --db-path results_moral_kimi-k2.5.db
```

### Digital Twin (BFI) Experiment

```bash
# Step 1: Estimate budget
bash scripts/run_digital_twin.sh openrouter minimax-m2 2000

# Step 2: Run experiment
bash scripts/run_digital_twin.sh openrouter minimax-m2 2000 --run

# Or run Python directly:
python -m experiments.digital_twin \
    --provider openrouter \
    --model minimax-m2 \
    --limit 2000 \
    --use-sampled-personas \
    --personas-file data/sampled_personas_2000.json \
    --db-path results_replicate_humans_minimax-m2_bfi.db
```

### Self-Introduction Collection

```bash
# OpenRouter (all registered models)
bash scripts/run_self_introduction.sh openrouter

# Specific models
bash scripts/run_self_introduction.sh openrouter kimi-k2.5 minimax-m2

# Local vLLM
bash scripts/run_self_introduction.sh vllm Qwen/Qwen3-4B-Instruct-2507
```

### Using Local Models with vLLM

```bash
# Start vLLM server
bash scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct 8000

# In another terminal, run experiments
export VLLM_BASE_URL="http://localhost:8000/v1"
bash scripts/run_moral_reasoning.sh vllm meta-llama/Llama-3.1-8B-Instruct --run
```

## Advanced Options

### Thinking Model Support

For models that produce extended chain-of-thought reasoning (e.g., DeepSeek-R1), enable thinking-model mode. This stores raw outputs and uses an LLM judge to extract numeric ratings:

```bash
THINKING_MODEL=true \
THINKING_MAX_TOKENS=16384 \
JUDGE_MODEL=gpt-5-mini \
JUDGE_PROVIDER=openai \
bash scripts/run_moral_reasoning.sh vllm deepseek-r1 --run
```

### Max Tokens

Control the max output tokens for non-thinking mode:

```bash
# Moral reasoning (default: 512)
MAX_TOKENS=1024 bash scripts/run_moral_reasoning.sh openrouter minimax-m2 --run

# Digital twin (default: 100)
MAX_TOKENS=200 bash scripts/run_digital_twin.sh openrouter minimax-m2 2000 --run
```

### Flagged Personas

Personas identified as logically inconsistent (e.g., a child persona who is married) can be excluded from API calls to save cost while preserving the full dataset structure. Experiment scripts automatically detect `data/flagged_personas.pkl` and inject dummy database entries for flagged personas.

To generate the flagged personas file:

```bash
python -m analysis.check_persona_inconsistency
```

This scans `data/sampled_personas_2000.json` and writes `data/flagged_personas.pkl`. Use `--dry-run` to preview without saving.

### Resumability

All experiments use SQLite databases for incremental saving. If a run is interrupted, simply re-run the same command -- completed tasks are automatically skipped.

## Analysis

### Step 1: Export Results to CSV

```bash
python analysis/extract_csv.py
```

This exports all SQLite result databases to CSV files under `csv_exports/`.

### Step 2: Run the Analysis Pipeline

The main analysis pipeline computes all metrics reported in the paper (BFI diagnostics, moral reasoning metrics, self-introduction linguistic analysis):

```bash
python -m analysis.analysis_pipeline \
    --data_dir ./csv_exports \
    --selfintro_dir ./self_introduction_results \
    --human_ref ./data/human_reference/wave_1_numbers.csv \
    --output_dir ./results
```

This produces:
- `results/analysis.json` -- all BFI + moral metrics per model
- `results/selfintro/` -- self-introduction linguistic features, mention rates, ICC, etc.
- `results/figures/` -- Coverage-vs-LID and Fidelity-vs-d plots (PDF)

Optional flags:
- `--flagged ./data/flagged_personas.pkl` -- exclude inconsistent personas
- `--skip_selfintro` -- skip self-introduction analysis
- `--skip_figures` -- skip figure generation

### Standalone Visualizations

```bash
# 3D metric illustrations (coverage, uniformity, complexity)
python analysis/visualizations/metrics_3d.py

# Uniformity figures
python analysis/visualizations/uniformity.py
```

See `analysis/README.md` for the full list of analysis scripts and detailed documentation.

## Citation

If you use this code or data, please cite our paper:

```bibtex
@article{xiao2026collapse,
  title={The Chameleon’s Limit: Investigating Persona Collapse and Homogenization in Large Language Models},
  author={Yunze Xiao, Vivienne Zhang, Chenghao Yang, Ningshan Ma, Weihao Xuan, Jen-tse Huang},
  journal={arXiv preprint arXiv:[to appear]},
  year={2026},
  note={arXiv ID pending}
}
```

## License

[To be specified]
