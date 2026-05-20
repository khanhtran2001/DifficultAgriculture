# DifficultyAgri

MIS-based diagnostics for agricultural object detection, with controlled copy-paste augmentation experiments across MinneApple, GWHD 2021, and WGISD.

## What This Project Does

This repository studies a practical question:

Can we predict in advance when copy-paste augmentation will help object detection, and when it will fail?

The core idea is **MIS (MinImageScorer)**, a scoring method that ranks object/image difficulty from baseline detector behavior (misses, localization quality, and false positives).

## Key Findings (Current)

- MIS correlates well with real detection failure signals.
- On structurally hard regimes (sub-resolution objects or strong domain shift), copy-paste is often ineffective or harmful.
- On datasets without those bottlenecks, copy-paste can still provide gains.

### Reported Quantitative Highlights

- Image-level correlation between MIS and miss-rate:
	- MinneApple: `r = 0.66`
	- GWHD: `r = 0.75`
	- WGISD: `r = 0.63`
- MinneApple size-level validity:
	- Spearman between object difficulty and AP-by-size rank: `r_s = -0.82`
	- Top difficulty tertile is dominated by Very Small objects (`64.4%`, AP `0.071`)
- GWHD domain-level validity:
	- Domain AP vs median difficulty anticorrelation: `r = -0.90`
	- Top-20% hard images show near-zero-AP object enrichment (`1.34x`, `p < 0.001`)

Source: `docs/Analyze/main_v2.tex` and generated experiment artifacts under `results/`.

## Important Experimental Results (From Paper)

Source: `docs/Analyze/main_v2.tex`

### 1) MinneApple Size-Level Validation (Table)

| Category | AP | Mean S_obj | Top difficulty tertile |
|---|---:|---:|---:|
| Very Small (<=400 px^2) | 0.071 | 0.152 | 64.4% |
| Small (400-1024 px^2) | 0.270 | 0.076 | 33.9% |
| Medium (>1024 px^2) | 0.493 | 0.053 | 1.7% |

Interpretation: hardest objects are concentrated in the very-small bucket, where AP is near zero.

### 2) Copy-Paste Outcomes Across Datasets (3-seed mean)

| Dataset | Near-zero AP concentration | Baseline AP | Best CP (Delta AP) | Worst CP (Delta AP) |
|---|---|---|---|---|
| MinneApple | Strong | 0.353 +/- 0.005 | -0.003 (Hi) | -0.008 (Rnd) |
| GWHD 2021 | Moderate | 0.229 +/- 0.013 | -0.006 (Hi) | -0.034 (Lo) |
| WGISD | Absent | 0.480 +/- 0.007 | +0.009 (Med) | +0.001 (Hi) |

Interpretation: copy-paste is negative on structurally flagged datasets (MinneApple, GWHD) and positive on unflagged WGISD.

### 3) GWHD domain difficulty vs AP

![GWHD domain score](docs/Analyze/gwhd_domain_score.png)

Interpretation: domains with higher median MIS generally have lower AP.

## Repository Layout

```text
DifficultyAgri/
	dagri/                  # Core library (data, baseline, scoring, augmentation, utils)
	configs/                # Dataset/experiment YAML configs
	experiments/            # Reproducible experiment entry points
	datasets/               # Dataset roots and formats
	notebooks/              # Analysis and reflection notebooks
	results/                # Experiment outputs and summaries
	docs/                   # Papers, reports, and figures
	tests/                  # Test files
```

## Setup

### Requirements

- Python `>= 3.13`
- Key packages: `torch`, `torchvision`, `ultralytics`, `pycocotools`, `matplotlib`, `seaborn`, `scikit-learn`

### Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
```

If editable install is not desired:

```bash
pip install .
```

## Quick Reproducibility Commands

Run from repository root.

### 1) Baseline training only

```bash
python experiments/01_only_training.py
```

### 2) Scoring only (using trained baseline artifacts)

```bash
python experiments/02_only_scoring.py
```

### 3) Train + scoring pipeline

```bash
python experiments/03_train_and_scoring.py
```

### 4) Multi-seed copy-paste experiment (example)

```bash
python experiments/055_multiseed_copy_paste_exp.py --seeds 123,456,789
```

## Notebook Workflow (How to Run)

Run from repository root.

### 1) Start Jupyter

```bash
source venv/bin/activate
jupyter lab
```

If `jupyter` is not installed in your environment:

```bash
source venv/bin/activate
pip install jupyterlab ipykernel
python -m ipykernel install --user --name difficultyagri --display-name "DifficultyAgri"
jupyter lab
```

### 2) Open a notebook

Recommended analysis notebooks:

- [notebooks/gwhd_2021_score_ap_size_inspection.ipynb](notebooks/gwhd_2021_score_ap_size_inspection.ipynb)
- [notebooks/reflection/augmented_dataset_analysis.ipynb](notebooks/reflection/augmented_dataset_analysis.ipynb)

### 3) Run notebook cells

- Select the `DifficultyAgri` kernel (or your project venv kernel).
- Run all cells from top to bottom to ensure paths and variables are initialized.
- For long cells, wait until execution completes before running the next one.

## Where to See Results

### Notebook outputs

- Tables and plots appear directly in notebook outputs.
- Any saved figures depend on notebook cells and are usually written under `docs/` or `results/`.

### Experiment result files

- [results/](results/) : root folder for run artifacts
- `results/*/Step_2_Train_and_Evaluate_BASELINE_MODEL/evaluation_results.json`
- `results/*/Step_3_Scoring_Dataset/score_results.json`
- `results/*/summary_augmentation_comparison.json`

### Paper/report outputs

- [docs/Analyze/main_v2.pdf](docs/Analyze/main_v2.pdf)
- [docs/GCCE_2026/](docs/GCCE_2026/)

