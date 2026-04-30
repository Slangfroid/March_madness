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
