import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="About Model - Tunisia Cars",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #E30613;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E30613;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
    .metric-box {
        background: linear-gradient(135deg, #E30613 0%, #C70039 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load model artifact"""
    model_path = Path(__file__).parent.parent.parent / "models" / "linear_regression_tunisia_cars.pkl"
    
    if not model_path.exists():
        return None
    
    try:
        artifact = joblib.load(model_path)
        return artifact
    except:
        return None

# Load dataset info
@st.cache_data
def load_dataset_info():
    """Load dataset metadata"""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "tunisia_cars_dataset.csv"
    
    if not data_path.exists():
        return None
    
    try:
        df = pd.read_csv(data_path)
        return {
            'total_rows': len(df),
            'num_features': len(df.columns),
            'brands': df['brand'].nunique(),
            'models': df['model'].nunique(),
            'year_range': f"{df['year'].min()} - {df['year'].max()}"
        }
    except:
        return None

artifact = load_model()
dataset_info = load_dataset_info()

# Header
st.markdown('<div class="main-header">ℹ️ About the Model</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Understanding the ML model behind price predictions</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <strong>🎯 Purpose:</strong> Explains the ML model used to predict car prices.
</div>
""", unsafe_allow_html=True)

# Model Performance Metrics
st.markdown("---")
st.subheader("📊 Model Performance")

if artifact and artifact.get('metrics'):
    col1, col2, col3 = st.columns(3)
    
    metrics = artifact['metrics']
    r2_score = metrics.get('r2_test', 0)
    mae = metrics.get('mae', 0)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{r2_score*100:.2f}%</div>
            <div class="metric-label">Accuracy (R² Score)</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("The model explains 89.91% of price variation")
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value">{mae:,.0f}</div>
            <div class="metric-label">Average Error (MAE)</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption("Predictions are off by ±3,876 TND on average")
    
    with col3:
        if dataset_info:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{dataset_info['total_rows']:,}</div>
                <div class="metric-label">Training Examples</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Model trained on 60,000+ car listings")
else:
    st.warning("⚠️ Model artifact not found. Please train the model first.")

# What is R² and MAE?
with st.expander("❓ What do these metrics mean?"):
    st.markdown("""
    **R² Score (Coefficient of Determination)**
    - Measures how well the model explains price variations
    - Range: 0% to 100% (higher is better)
    - **89.91%** means the model captures most price patterns
    - Remaining ~10% is due to random factors or unmeasured features
    
    **MAE (Mean Absolute Error)**
    - Average difference between predicted and actual prices
    - Measured in TND (Tunisian Dinar)
    - **3,876 TND** means predictions are typically within ±4,000 TND
    - For a 40,000 TND car, expect ±10% error
    """)

# Model Algorithm
st.markdown("---")
st.subheader("🧠 Algorithm: Multiple Linear Regression")

col_algo1, col_algo2 = st.columns(2)

with col_algo1:
    st.markdown("""
    ##### What is Linear Regression?
    
    Simple ML algorithm that predicts price based on features (brand, year, mileage, etc.).
    
    **How it works:**
    1. Learns from 60,000+ car listings
    2. Identifies which features affect price
    3. Creates formula to estimate prices
    4. Applies formula to your car specs
    """)

with col_algo2:
    st.markdown("""
    ##### Why Linear Regression?
    
    **Advantages:**
    - Easy to understand
    - Fast predictions
    - Transparent
    - Reliable for price estimation
    
    **Trade-offs:**
    - Assumes linear relationships
    - Limited to simpler patterns
    
    **Best for:** Educational projects and interpretability.
    """)

# Features Used
st.markdown("---")
st.subheader("🔑 Features Used by the Model")

col_feat1, col_feat2, col_feat3 = st.columns(3)

with col_feat1:
    st.markdown("""
    ##### 📅 Basic Features
    - **Year** → Age of car
    - **Mileage** → Distance traveled
    - **Brand** → Manufacturer
    - **Model** → Specific car model
    - **Fuel Type** → Essence/Diesel/Hybrid
    - **Gearbox** → Manual/Automatic
    """)

with col_feat2:
    st.markdown("""
    ##### 🔧 Technical Features
    - **Body Type** → Sedan/SUV/Hatchback
    - **Horsepower** → Engine power
    - **Engine Size** → Liters (e.g., 1.6L)
    - **Condition** → Excellent to Repair needed
    - **Origin** → Local or Imported
    """)

with col_feat3:
    st.markdown("""
    ##### 📍 Other Features
    - **Location** → City/Governorate
    - **Color** → Exterior color
    - **Owners** → Number of previous owners
    - **Accident History** → Yes/No
    - **Optional Features** → GPS, Sunroof, etc.
    """)

# Engineered Features
st.markdown("---")
st.subheader("⚙️ Engineered Features")

col_eng1, col_eng2 = st.columns(2)

with col_eng1:
    st.markdown("""
    **Car Age**
    ```python
    age = 2025 - year
    ```
    Example: 2020 car → 5 years old
    """)

with col_eng2:
    st.markdown("""
    **Mileage Per Year**
    ```python
    mileage_per_year = mileage / age
    ```
    Example: 60,000 km / 5 years = 12,000 km/year
    """)

# Training Process
st.markdown("---")
st.subheader("🎓 How Was the Model Trained?")

col_train1, col_train2 = st.columns([2, 1])

with col_train1:
    st.markdown("""
    **Training Pipeline (7 Steps):**
    
    1. **Data Loading** → Load 60,000+ car listings from CSV
    2. **Exploratory Analysis** → Understand price distributions and patterns
    3. **Data Cleaning** → Handle missing values (median for numbers, drop incomplete rows)
    4. **Feature Engineering** → Create `age` and `mileage_per_year`
    5. **Encoding** → Convert categorical features (brand, fuel, etc.) to numbers
    6. **Training** → Fit Linear Regression model on 80% of data
    7. **Evaluation** → Test on remaining 20% to measure accuracy
    
    **Result:** Model saved as `linear_regression_tunisia_cars.pkl`
    """)

with col_train2:
    if dataset_info:
        st.markdown("##### 📈 Dataset Stats")
        st.markdown(f"""
        - **Total Listings:** {dataset_info['total_rows']:,}
        - **Brands:** {dataset_info['brands']}
        - **Models:** {dataset_info['models']}
        - **Year Range:** {dataset_info['year_range']}
        - **Split:** 80% train / 20% test
        """)

# Data Source
st.markdown("---")
st.subheader("📦 Data Source")

col_data1, col_data2 = st.columns(2)

with col_data1:
    st.markdown("""
    ##### Synthetic Dataset
    
    Training data is **synthetic** (artificially generated) but mimics real Tunisian market patterns.
    
    **Key factors:**
    - Brand base prices
    - Age depreciation (4%/year)
    - Mileage penalty
    - Condition adjustments
    - Location premiums
    """)

with col_data2:
    st.markdown("""
    ##### ⚠️ Disclaimer
    
    **Educational Purpose Only**
    
    This model is for learning, not for official valuations or commercial use.
    
    Always consult real market data for actual pricing.
    """)

# Model Limitations
st.markdown("---")
st.subheader("⚠️ Limitations")

col_limit1, col_limit2 = st.columns(2)

with col_limit1:
    st.markdown("""
    **Data:**
    - Synthetic (not real market)
    - Limited to training period
    - No seasonal variations
    """)

with col_limit2:
    st.markdown("""
    **Model:**
    - Linear assumptions
    - ±3,876 TND average error
    - Larger errors for rare cars
    """)

# Future Improvements
st.markdown("---")
st.subheader("🚀 Possible Improvements")

col_imp1, col_imp2 = st.columns(2)

with col_imp1:
    st.markdown("""
    **Model:**
    - Try advanced algorithms (Random Forest, XGBoost)
    - Add confidence intervals
    - Cross-validation
    """)

with col_imp2:
    st.markdown("""
    **Data:**
    - Collect real market data
    - Add temporal features
    - Include market trends
    """)

# Technical Stack
st.markdown("---")
st.subheader("🛠️ Technical Stack")

col_tech1, col_tech2, col_tech3 = st.columns(3)

with col_tech1:
    st.markdown("""
    ##### 💻 ML & Data
    - **Python** 3.10+
    - **pandas** (data manipulation)
    - **NumPy** (numerical computing)
    - **scikit-learn** (ML algorithms)
    - **Joblib** (model persistence)
    """)

with col_tech2:
    st.markdown("""
    ##### 📊 Visualization
    - **Matplotlib** (charts)
    - **Seaborn** (statistical plots)
    - **Plotly** (interactive charts)
    """)

with col_tech3:
    st.markdown("""
    ##### 🌐 Deployment
    - **Streamlit** (web app)
    - **Streamlit Cloud** (hosting)
    - **Git/GitHub** (version control)
    """)

# Team
st.markdown("---")
st.subheader("👥 Project Team")

st.markdown("""
<div class="info-card">
    <strong>Developed by:</strong><br>
    Khalil Amamri | Montassar Zreilli | Wassim Mnassri | Mahdi Hadj Amor
    <br><br>
    <strong>Project Type:</strong> Educational Machine Learning Project<br>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>ℹ️ About the Model | Tunisia Car Price Predictor</p>
    <p>For educational purposes - Demonstrating ML lifecycle from data to deployment</p>
</div>
""", unsafe_allow_html=True)
