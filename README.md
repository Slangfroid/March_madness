# Predicting Unpredictability: NCAA March Madness Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![Databricks](https://img.shields.io/badge/Platform-Databricks-red)
![Accuracy](https://img.shields.io/badge/Accuracy-66%25-brightgreen)

## Overview
A machine learning model built to predict the outcomes of the 2026 NCAA March Madness 
Tournament. This project was completed as a Data Science Senior Project at Michigan State 
University — Spring 2026.

The goal wasn't just to predict basketball games. It was to confront a real fear: 
*what happens when AI takes your job?* Building this model showed me that the real 
risk isn't AI replacing you — it's not learning how to build with it.

---

## Results
- **66% accuracy** — 41 of 63 tournament games predicted correctly
- **Top 2-3%** of ESPN bracket challenge during the tournament
- **Predicted Final Four:** Duke, Houston, Michigan, Arizona
- **Predicted National Champion:** Arizona

---

## Data Sources
- **[Kaggle NCAA March Madness Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)**
  - `MNCAATourneyCompactResults.csv` — historical tournament game results (2003-2025)
  - `MNCAATourneySeeds.csv` — tournament seeds for all seasons
  - `MMasseyOrdinals.csv` — 197 ranking systems including POM, SAG, RPI, NET
  - `MTeams.csv` — team ID mappings
  - `MTeamSpellings.csv` — alternative team name spellings
  - `SampleSubmissionStage2.csv` — submission template

- **[Barttorvik T-Rank](https://barttorvik.com/2026_team_results.csv)**
  - 2026 current season efficiency stats for all 365 Division I teams
  - Downloaded directly as CSV: `barttorvik.com/2026_team_results.csv`

---

## How It Works

### The Core Idea — Pairwise Differences
Instead of feeding the model one team's stats, every matchup is represented as the 
*difference* between two teams' stats:

```python
diff_AdjEM      = Team_A_AdjEM - Team_B_AdjEM        # e.g. 37.4 - 28.1 = +9.3
diff_SeedNum    = Team_A_Seed  - Team_B_Seed          # e.g. 1 - 2 = -1
diff_Massey_POM = Team_A_POM   - Team_B_POM           # e.g. 1 - 8 = -7
```

This gives the model 30 difference features per matchup and directly answers 
the question: *how much better is Team A than Team B?*

### Features (30 total)

**Efficiency Ratings**
- `AdjEM` — Adjusted Efficiency Margin (points per 100 possessions vs avg opponent)
- `AdjOE` — Adjusted Offensive Efficiency
- `AdjDE` — Adjusted Defensive Efficiency  
- `AdjTempo` — Adjusted possessions per 40 minutes
- `Net Rating` — Overall net rating proxy

**Shooting Stats**
- `eFGPct`, `FG3Pct`, `FG3Rate` — offensive shooting metrics
- `OppFG3Pct`, `OppBlockPct`, `OppStlRate` — defensive metrics
- `FTRate`, `TOPct`, `StlRate` — possession efficiency

**Physical/Experience**
- `AvgHeight`, `EffectiveHeight` — roster size metrics
- `Experience` — average years of college experience
- `Active Coaching Length Index` — coach tournament experience

**Massey Rankings (9 systems)**
- `POM` (KenPom), `SAG` (Sagarin), `RPI`, `MOR` (Morningstar)
- `NET`, `BPI`, `DOL`, `KPI`, `WLK`

**Tournament Context**
- `SeedNum` — tournament seed (1-16)

### Training Data
- **1,419 real NCAA tournament games** from 2003-2025
- Label = 1 if lower TeamID team won, 0 if higher TeamID team won
- Features = 30 stat differences between the two teams

### Model
Two models were trained and compared using 5-fold cross validation:

| Model | CV Log Loss |
|-------|-------------|
| Logistic Regression | 0.5305 ± 0.0267 |
| **XGBoost** | **0.5265 ± 0.0416** |

XGBoost was selected as the final model.

---

## Project Structure
march-madness-ml/
│
├── data/
│   ├── march_madness_post.csv      # PST historical stats + tournament outcomes
│   ├── march_madness.csv           # Regular season game logs
│   ├── 2026_team_results.csv       # Barttorvik 2026 stats
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   ├── MMasseyOrdinals.csv
│   ├── MTeams.csv
│   ├── MTeamSpellings.csv
│   └── SampleSubmissionStage2.csv
│
├── notebooks/
│   └── march_madness_predictor.py  # Main Databricks notebook
│
├── outputs/
│   ├── shap_importance.png         # Feature importance chart
│   ├── shap_beeswarm.png           # SHAP beeswarm plot
│   └── shap_Arizona_vs_Michigan.png # Waterfall plot example
│
└── README.md

---

## How to Run

### Prerequisites
```python
%pip install rapidfuzz xgboost scikit-learn shap
```

### Steps
1. Download Kaggle competition data from the 
   [NCAA March Madness Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
2. Download 2026 Barttorvik data:
http://barttorvik.com/2026_team_results.csv

3. Upload all CSV files to Databricks
4. Run `march_madness_predictor.py` top to bottom
5. Output saved to `submission_2026.csv`

---

## Key Findings

### Most Important Features (SHAP)
1. **AdjEM** — single best predictor of team quality
2. **Net Rating** — correlated with AdjEM, confirms efficiency matters most
3. **AdjOE** — offensive efficiency is more predictive than defensive
4. **RPI** — despite being old-fashioned, still carries signal
5. **POM** — KenPom ratings independently confirm efficiency ratings
6. **SeedNum** — surprisingly low importance since AdjEM already captures seed quality

### Notable 2026 Predictions
| Matchup | Prediction | Probability |
|---------|-----------|-------------|
| Duke vs Connecticut (Elite 8) | Duke | 73% |
| Houston vs Florida (Elite 8) | Houston | 54% |
| Arizona vs Michigan (Final Four) | Michigan | 76% |
| Duke vs Arizona (Championship) | Arizona | 51% |

### Known Limitations
- Most 2026 teams defaulted to 2025 season averages for non-efficiency features
- Only 9 of 197 Massey systems had 2026 data available
- AdjEM and Net Rating are correlated — effectively double-counting efficiency
- 1,419 training games is relatively small for a machine learning problem

---

## What I Would Improve Next Time
- Engineer 2026 efficiency stats directly from `MRegularSeasonDetailedResults.csv`
  to eliminate dependency on external data
- Train on regular season games between tournament-caliber teams to 10x training data
- Remove Net Rating to eliminate feature redundancy with AdjEM
- Add recent form features — last 10 games win % before tournament
- Use time-series cross validation to prevent data leakage between seasons
- Calibrate probabilities with `CalibratedClassifierCV` to improve log loss
- Add Vegas betting lines as a feature — closing spreads are highly predictive

---

## Tools & Technologies
- **Python** — Polars, Pandas, NumPy
- **Machine Learning** — XGBoost, Scikit-learn
- **Explainability** — SHAP
- **Platform** — Databricks
- **Data** — Kaggle, Barttorvik
- **AI Assistance** — [Claude AI](https://claude.ai) by Anthropic

---

## AI Assistance
This project was built with significant help from **Claude AI** (Anthropic). Claude 
assisted with:
- Debugging data pipeline issues (fuzzy name matching, TeamID mapping)
- Writing the pairwise feature engineering approach
- Fixing the bracket simulation logic
- Generating SHAP analysis code
- Explaining what each feature means and how the model works

The experience of building this with Claude directly informed the personal insight 
at the center of this project: AI doesn't replace you — it makes you a more 
powerful creator. The only limit is yourself.

---

## Personal Reflection
For a while I feared what AI meant for my career as a graduating data science student. 
Building this model gave me two key insights:

1. **AI can't predict humanness** — March Madness upsets happen because humans are 
   unpredictable. No model will ever generate a perfect bracket and that's okay. 
   It means human judgment will always matter.

2. **AI makes you a creator** — I built something I couldn't have built alone. 
   That's not inadequacy — that's leverage. The tools have changed but the 
   thinking still has to come from you.

---

## Author
**Nathaniel Leonardson**  
Data Science Senior — Michigan State University  
Senior Project — Spring 2026

---

## License
MIT License — feel free to use, modify, and build on this for your own 
March Madness predictions!
