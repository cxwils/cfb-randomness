# CFB Upset Randomness

This project examines the randomness of college football upsets through 
statistical analysis and predictive modeling. Historical FBS game data is 
used to identify factors associated with upsets, evaluate prediction 
performance, and quantify the extent to which seemingly unpredictable 
outcomes can be explained by measurable variables.

## Setup

1. Clone the repo
2. Create a virtual environment: `py -3.13 -m venv venv`
3. Activate it: `.\venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Install dependencies: `pip install -r requirements.txt`
5. Add your CollegeFootballData API key to a `.env` file: `CFBD_API_KEY=your_key_here`

## Reproducing the Analysis

Run in this order:

1. `python src/cfb_upsets/data_acquisition.py` — pulls raw game and betting line data (2021-2025) from the CFBD API into `data/raw/`
2. `python src/cfb_upsets/cleaning.py` — filters to FBS-vs-FBS games, joins games with betting lines, outputs clean CSVs to `data/processed/`
3. `notebooks/01_eda.ipynb` — exploratory analysis: upset rates by spread, season, conference, home/away status
4. `notebooks/02_feature_exploration.ipynb` — feature engineering (Elo differential, rolling win %, etc.)
5. `notebooks/03_statistical_analysis.ipynb` — chi-square tests and multivariate logistic regression
6. `notebooks/04_modeling.ipynb` — predictive modeling, evaluation (Brier score, calibration), and Monte Carlo simulation
7. `notebooks/05_presentation_demo.ipynb` — condensed walkthrough of key findings (used for final presentation)

All figures referenced in the report are saved to `reports/figures/`.

## Project Structure

[brief folder tree]

## Key Findings

- Overall upset rate: 27.4% (2021-2025)
- Upset rate strongly correlates with betting spread magnitude (46.5% for narrow spreads vs. 4.6% for wide spreads)
- Home underdogs upset significantly more than away underdogs (31.1% vs. 25.1%, p=0.0001), though this effect is not independently significant once controlling for spread
- A full predictive model (spread, Elo, recent form, conference, home/away) does not meaningfully outperform the spread alone (Brier score 0.176 vs. 0.175)
- Monte Carlo simulation confirms real-world upset variance is statistically consistent with the model's predicted probabilities, suggesting the majority of residual uncertainty is irreducible rather than a modeling gap
