# NFL 4th-Down WinPct Recommender

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
  <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Compact multi-task MLP that recommends **go / field goal / punt** on 4th down by predicting **ΔWP** (change in win probability) for each action. Trained on 2016–2024 NFL play-by-play with labels from `nfl4th`. Includes EDA, reproducible data pipeline, and training notebooks.

**Headline test results (2023–2024):**
- **Policy agreement:** `0.794`
- **Mean regret (ΔWP):** `0.0023` (~0.23 pp)  
- **Late & close (<600s, |score| ≤ 8):** agreement `0.744`, regret `0.0099` (~0.99 pp)

---

## What this project does

- Builds a **clean 4th-down state dataset** from pbp.
- Uses `nfl4th` to compute **action-conditional WP** and derives **ΔWP** targets.
- Trains a **multi-task MLP** (main head: 3 ΔWP targets; aux heads: FG-make and 4th-down conversion).
- Evaluates with **policy agreement** and **mean regret**, overall and by **yardline / yards-to-go** bins.

---

## Quickstart

### 1) Environment

```bash
# Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# R (>= 4.2) – one time
R -q -e 'install.packages(c("tidyverse"))'
R -q -e 'if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes")'
R -q -e 'remotes::install_github("guga31bb/nfl4th")'
```

### 2) Raw data

Place nflfastR play-by-play parquet files under `data/raw/`, e.g.:

```
data/raw/pbp_2016_2024.parquet
# or multiple files:
# data/raw/pbp_2016.parquet ... data/raw/pbp_2024.parquet
```

### 3) Build features and labels

```bash
# Extract 4th-down states → data/interim/fourthdown_states.csv
python -m nfl_4thdown_dwp.features

# Create labels with nfl4th (wp_go/wp_fg/wp_punt) and compute ΔWP → data/processed/4thdown_labeled.csv
Rscript nfl_4thdown_dwp/labels/build_labels.R
```

### 4) Explore & train

Open the notebooks in `notebooks/` for:
- **EDA** (sanity checks, monotonicity, bins)
- **Training** (multi-task MLP) and **evaluation** (agreement / regret + per-bin plots)

Trained weights and predictions are saved to `models/` and figures to `reports/figures/`.

---

## Model card (brief)

- **Inputs (examples)**  
  `quarter, game_seconds_remaining, half_seconds_remaining, yardline_100, ydstogo, score_differential, timeouts_off, timeouts_def, temp_f, wind_mph, spread_line, receive_2h_ko, fg_dist_yd, in_fg_range, one_score, late_game, second_half, timeouts_total, timeouts_diff, roof(one-hot), surface(one-hot)`

- **Outputs**  
  `ΔWP_go, ΔWP_fg, ΔWP_punt` (main head), plus `p_fg_make`, `p_go_convert` (aux heads)

- **Recommendation**  
  `argmax_a ΔWP_a` with optional “toss-up” flag when (best − second best) < 0.5 pp

- **Training split**  
  Train: 2016–2021, Val: 2022, Test: 2023–2024

- **Loss/opt**  
  Huber(ΔWP) + 0.2·MSE(aux); AdamW, BN, Dropout, L2

---

## Reproducibility

- Deterministic data steps (`features.py`, `labels/build_labels.R`)
- Temporal split to avoid leakage
- Seeds set in notebooks for NumPy/TensorFlow
- Artifacts:
  ```
  models/mlp_dwp_plus_aux_best.keras
  models/preds_test.parquet
  reports/figures/*.png
  ```

---

## Project organization

```
├── LICENSE
├── Makefile
├── README.md
├── data
│   ├── raw/         # nflfastR pbp parquet files
│   ├── interim/     # 4th-down states (CSV/Parquet)
│   └── processed/   # labeled ΔWP dataset
├── docs/
├── models/          # trained weights & predictions
├── notebooks/       # EDA, training, evaluation
├── references/
├── reports/
│   └── figures/     # generated plots
├── requirements.txt
├── pyproject.toml
├── setup.cfg
└── nfl_4thdown_dwp/
    ├── __init__.py
    ├── features.py          # build 4th-down states
    ├── labels/
    │   └── build_labels.R   # add nfl4th WP & compute ΔWP
    ├── modeling/
    │   ├── __init__.py
    │   ├── train.py         # (optional) scripted training
    │   └── predict.py       # (optional) batch inference
    └── plots.py
```

---

## Notes & caveats

- `wp_punt` can be structurally missing in short field; this repo imputes a **small constant floor** and documents the choice.
- Weather for closed roofs is set to `temp=70°F`, `wind=0 mph`; open-air uses open-game medians.
- This model mirrors `nfl4th`’s teacher labels; it’s a decision assistant, not a replacement for domain judgment.

---

## License

See [`LICENSE`](LICENSE) for details.
