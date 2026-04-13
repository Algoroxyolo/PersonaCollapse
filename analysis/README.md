# Analysis

This folder contains analysis scripts for post-processing experiment results from persona-collapse experiments.

## Main Analysis Pipeline

The primary analysis script is `analysis_pipeline.py`, which runs the full suite of metrics reported in the paper.

### Quick Start

From the **repository root**:

```bash
python -m analysis.analysis_pipeline \
    --data_dir ./csv_exports \
    --selfintro_dir ./self_introduction_results \
    --human_ref ./data/human_reference/wave_1_numbers.csv \
    --output_dir ./results
```

Or equivalently:

```bash
python analysis/analysis_pipeline.py \
    --data_dir ./csv_exports \
    --selfintro_dir ./self_introduction_results \
    --output_dir ./results
```

(`--human_ref` defaults to `./data/human_reference/wave_1_numbers.csv`)

### Prerequisites

The pipeline expects CSV-exported experiment results. To generate them from SQLite databases:

```bash
python analysis/extract_csv.py
```

This produces `csv_exports/` with subfolders like `digital_twin_<model>/` and `moral_reasoning_<model>/`, each containing a `*_response_matrix.csv`.

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--data_dir` | Yes | - | Directory containing `digital_twin_*` and `moral_reasoning_*` subfolders with CSV matrices |
| `--selfintro_dir` | No | `None` | Directory containing `introductions_*.jsonl` files |
| `--human_ref` | No | `./data/human_reference/wave_1_numbers.csv` | Path to human BFI reference data |
| `--flagged` | No | `None` | Path to `flagged_personas.pkl` for filtering inconsistent personas |
| `--output_dir` | No | `./results` | Output directory |
| `--skip_selfintro` | No | `false` | Skip self-introduction analysis |
| `--skip_figures` | No | `false` | Skip figure generation |

### Outputs

```
results/
├── analysis.json               - All BFI + Moral metrics per model
├── tables/                     - LaTeX tables (if generated)
├── figures/
│   └── fig_combined.pdf        - Coverage-vs-LID and Fidelity-vs-d plots
└── selfintro/
    ├── selfintro_features.csv  - Extracted linguistic features
    ├── layer1_mention_rates.csv
    ├── layer3_template_detection.json
    ├── layer4_feature_eta2.csv
    ├── layer4b_incremental_r2.csv
    └── layer4c_icc.csv
```

### What It Computes

**BFI + Moral Reasoning (per model):**
- Effective Likert levels (inverse Simpson diversity)
- Participation ratio (PCA dimensionality)
- Local Intrinsic Dimensionality (LID)
- Hopkins statistic (clustering tendency)
- Silhouette scores (K=5, 10)
- V-measure (demographic alignment)
- Density & Coverage (vs. human reference, BFI only)
- Political eta-squared (sensitivity to political orientation)
- Incremental R-squared (demographic explanatory power)
- Persona fidelity and Cohen's d (BFI only)

**Self-Introduction (per model):**
- Linguistic features (TTR, hapax ratio, Guiraud, hedging, sentiment)
- Attribute mention detection (country, age, gender, class, politics)
- Template/opening diversity analysis
- Feature eta-squared by demographic group
- Incremental R-squared for key features
- Intraclass Correlation Coefficient (ICC) for persona consistency

## Other Scripts

| Script | Description |
|--------|-------------|
| `extract_csv.py` | Extract persona x question/scenario response matrices from experiment SQLite databases into CSV files. |
| `check_persona_inconsistency.py` | Scan persona JSON for logically inconsistent attribute combinations and produce `data/flagged_personas.pkl`. |

## Visualizations

The `visualizations/` subfolder contains matplotlib scripts for generating figures:

| Script | Description |
|--------|-------------|
| `visualizations/metrics_3d.py` | 3D scatter plots illustrating coverage, uniformity, and complexity metrics. |
| `visualizations/uniformity.py` | 3D diagrams comparing human, over-regular, and clustered population distributions. |

## Adding New Scripts

Additional analysis scripts from collaborators can be added to this directory.
Place figure-generation scripts in `visualizations/` to keep the structure organized.
