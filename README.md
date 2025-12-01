
# 🚗 Tunisia Car Price Predictor

Lightweight Streamlit application and supporting notebook to predict used car prices in Tunisia. The project includes data, a training notebook, and a Streamlit UI that loads an exported model artifact to provide instant price estimates.

## Contents
- `app/app.py` — Streamlit application and user interface
- `data/raw/tunisia_cars_dataset.csv` — dataset used for EDA and training
- `notebooks/Tunisia_Cars_Price_Prediction.ipynb` — exploratory analysis, training, and model export
- `models/` — trained model artifact (expected file: `linear_regression_tunisia_cars.pkl`)
- `requirements.txt` — Python dependencies

## Quickstart (Windows)

1. Clone the repository and change to the project directory:

```powershell
cd D:\GITHUB\projet_voiture\Car_Valuation_Tunisia
```

2. Create a virtual environment and install dependencies

PowerShell (recommended) — allow local script execution once if blocked:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
& .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you prefer Command Prompt (no policy change needed):

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

3. Run the Streamlit app

```powershell
streamlit run app/app.py
```

The app will open in your browser (typically at http://localhost:8501).

## Notebook / Model training

- Open `notebooks/Tunisia_Cars_Price_Prediction.ipynb` to reproduce training, feature engineering, and model export. The notebook saves an artifact compatible with the Streamlit app at `models/linear_regression_tunisia_cars.pkl` using `joblib`.

Artifact shape (expected keys):
- `model` — trained scikit-learn estimator
- `scaler` — numeric scaler used at training time
- `feature_columns` — list of feature column names used by the model
- `numeric_columns` — list of numeric column names
- `categorical_columns` — list of original categorical columns
- `metrics` — evaluation metrics (optional)

If the app reports "Model file not found", run the notebook to generate the artifact or place a compatible artifact into the `models/` folder.

## Troubleshooting
- PowerShell blocks script execution: either set the execution policy for the current user (`RemoteSigned`) or use the Command Prompt activator (`.venv\Scripts\activate.bat`).
- Missing dependencies: run `pip install -r requirements.txt` in the activated venv.
- Model mismatch errors (feature columns changed): retrain and export the artifact or update the Streamlit input code to match the artifact's expectations.

## Development notes
- The app uses `st.cache_resource` / `st.cache_data` for model and dataset caching. When iterating on the model or dataset, restart the Streamlit server to clear caches.
- Keep large raw datasets and model binaries out of version control. See `.gitignore`.

## Contributing
- Issues and pull requests welcome. If you retrain the model, include training notes and export the final artifact to `models/` with a matching schema.

## Authors
Khalil Amamri • Montassar Zreilli • Wassim Mnassri • Mahdi Hadj Amor

## License
Add your project license here (e.g., MIT, Apache-2.0) — or create one if this repo will be shared publicly.
