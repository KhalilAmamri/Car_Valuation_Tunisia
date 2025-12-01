<div align="center">

# 🚗 Tunisia Car Price Predictor (Tunisia Car Valuation)

Predict realistic used car prices across all 24 Tunisian governorates using a Multiple Linear Regression model trained on a rich synthetic dataset (60,000+ listings, 23 brands, diverse vehicle attributes).

**Live App:** https://car-valuation-tunisia.streamlit.app/

</div>

---

## 📌 Table of Contents

1. Overview
2. Key Features
3. Data & Generation Process
4. Modeling Approach
5. Project Structure
6. Installation & Quick Start
7. Usage (CLI / Notebook / App)
8. Model Artifact Details
9. Metrics & Evaluation
10. Feature Definitions
11. How to Reproduce End‑to‑End
12. Roadmap & Possible Enhancements
13. Disclaimer

---

## 1. Overview

This project builds a transparent, reproducible workflow to estimate used car prices in Tunisia. It combines:

- A synthetic but domain‑aware dataset generator reflecting local market dynamics (brand, condition, fuel type, geography, etc.).
- A streamlined Jupyter notebook training pipeline (Sections 1–7) for data loading → EDA → cleaning → feature engineering → training → evaluation → artifact saving.
- A responsive Streamlit web app with dynamic dropdowns sourced directly from the dataset to ensure encoding consistency and correct categorical influence on predictions.

> Goal: Educational illustration of a structured ML lifecycle (data generation → modeling → deployment) rather than a real commercial valuation engine.

---

## 2. Key Features

- 🔧 Synthetic dataset with realistic adjustments (brand base price, body type premiums, condition, gearbox, fuel, location, color, accident & ownership penalties, options adders).
- 🧪 Multiple Linear Regression with engineered features (`age`, `mileage_per_year`).
- 🧼 Robust missing value handling (median for numeric, drop rows with missing critical categoricals).
- 🏷 Dynamic categorical encoding using `pandas.get_dummies(drop_first=True)` aligned between training and inference.
- 🖥 Streamlit UI providing instant predictions and model insight panels (top coefficients, feature importance proxy via weights).
- 🧩 Clean artifact (`joblib`) bundling model, scaler, columns, engineered feature names, and metrics (R² & MAE only—RMSE intentionally removed).
- 📊 Minimal, focused visualizations (scatter Year vs Price, Mileage vs Price; Actual vs Predicted evaluation scatter).

---

## 3. Data & Generation Process

Located in `scripts/script_to_generate_dataset.ipynb`.

Generation logic applies layered effects:

- Brand base prices + model adjustments (SUV / pickup / premium / economy patterns).
- Age depreciation (4% of base per year) & mileage penalty (≈12 TND per 1,000 km).
- Condition deltas (from +15% for excellent to −30% for repair needed).
- Gearbox (automatic +8%), fuel (hybrid +10%, diesel −3%).
- Body type multipliers (e.g., pickup 1.20×, SUV 1.15×).
- Location premiums (e.g., Tunis +5%).
- Color subtle boosts (neutral colors add small amounts).
- Accident and owner count penalties.
- Optional feature additive contributions (sunroof, alloy wheels, etc.).
- Additive stochastic noise for price variance.

All rows constrained to a final price range [5,000 – 250,000] TND.

> Synthetic nature enables experimentation without privacy concerns.

---

## 4. Modeling Approach

- Algorithm: Ordinary Least Squares via `sklearn.linear_model.LinearRegression`.
- Feature Space: Numeric features (scaled) + one‑hot encoded categoricals (drop_first to avoid dummy trap) + engineered features.
- Scaling: StandardScaler applied to numeric columns only (stored in artifact).
- Negative prediction safeguard: Predictions clamped to ≥ 0.

---

## 5. Project Structure

```
Car_Valuation_Tunisia/
├── app/                      # Streamlit application
├── data/
│   └── raw/                  # Generated dataset CSV
├── models/                   # Saved model artifacts (.pkl)
├── notebooks/
│   └── Tunisia_Cars_Price_Prediction.ipynb  # Training workflow
├── scripts/
│   └── script_to_generate_dataset.ipynb     # Dataset generation
├── images/                   # (Reserved for visuals)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 6. Installation & Quick Start

```pwsh
# Clone repository
git clone https://github.com/KhalilAmamri/Car_Valuation_Tunisia.git
cd Car_Valuation_Tunisia

# (Optional) Create virtual environment
python -m venv .venv
./.venv/Scripts/Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Usage

### A. Regenerate Dataset

Open `scripts/script_to_generate_dataset.ipynb` and run all cells (or export logic into a script if desired) to produce `data/raw/tunisia_cars_dataset.csv`.

### B. Train / Refresh Model

Run sections 1 → 7 of `notebooks/Tunisia_Cars_Price_Prediction.ipynb`. This will output a fresh artifact: `models/linear_regression_tunisia_cars.pkl`.

### C. Launch Streamlit App Locally

```pwsh
streamlit run app/app.py
```

Open the provided local URL (default: http://localhost:8501).

### D. Use Hosted Version

Visit: https://car-valuation-tunisia.streamlit.app/

---

## 8. Model Artifact Details

`models/linear_regression_tunisia_cars.pkl` includes:

```python
{
	'model_name': 'Linear Regression',
	'model': <LinearRegression>,
	'scaler': <StandardScaler>,
	'numeric_columns': [...],
	'feature_columns': [... all columns after encoding ...],
	'categorical_columns': ['brand','model','fuel','gearbox','vehicle_condition','car_body','import_or_local','location','color'],
	'engineered_features': ['age','mileage_per_year'],
	'metrics': {
		 'r2_test': <float>,
		 'mae': <float>
	}
}
```

RMSE intentionally excluded to streamline evaluation focus.

---

## 9. Metrics & Evaluation

Displayed in notebook Section 6:

- R² Score (variance explained)
- MAE (average absolute error in TND)

Scatter plot: Actual vs Predicted provides visual alignment (45° reference line).

> Regenerating the dataset will change performance values slightly due to stochastic noise.

---

## 10. Feature Definitions (Selected)

- `age`: `2025 - year` (future anchor year for consistency).
- `mileage_per_year`: `mileage / max(age,1)`.
- One‑hot encoded categorical fields: brand, model, fuel, gearbox, vehicle_condition, car_body, import_or_local, location, color.
- Options (binary): air_conditioning, parking_sensor, rear_camera, sunroof, alloy_wheels, bluetooth, gps (currently not all used directly in baseline model—extendable).

---

## 11. How to Reproduce End‑to‑End

1. Generate dataset (script notebook).
2. Inspect initial scatter plots (Year vs Price, Mileage vs Price).
3. Clean missing values (median numeric, drop incomplete categorical rows).
4. Engineer `age` & `mileage_per_year`.
5. Encode categoricals (`get_dummies(drop_first=True)`).
6. Split, scale numerics, train linear regression.
7. Evaluate (R², MAE + scatter plot).
8. Save artifact to `models/`.
9. Launch Streamlit app which loads artifact and dataset for dropdowns.

---

## 12. Roadmap & Possible Enhancements

- Add regularization (Ridge/Lasso) & compare generalization.
- Introduce feature importance via permutation tests.
- Persist prediction logs (user inputs → predicted price) for analytics.
- Add confidence intervals around predictions.
- Switch synthetic year anchor from 2025 to dynamic current year.
- Replace categorical row drops with mode imputation for higher data retention.
- Containerize (Dockerfile + devcontainer updates).
- CI workflow (lint, test training pipeline on sample subset).

---

## 13. Disclaimer

This dataset is entirely synthetic and for educational/demonstrative use. Predictions do **not** represent official market valuations. Always consult real market data and professional sources for financial decisions.

---

## 🙌 Acknowledgements

Team Members: Khalil Amamri, Montassar Zreilli, Wassim Mnassri, Mahdi Hadj Amor.

Inspired by standard supervised ML workflows and open data modeling best practices.

---

## 📬 Contact / Support

Feel free to open an issue or propose improvements via pull requests.

---

Enjoy exploring Tunisian car valuation! 🚗🇹🇳
