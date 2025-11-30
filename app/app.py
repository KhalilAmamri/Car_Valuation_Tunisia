import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Tunisia Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .prediction-box {
        background: linear-gradient(135deg, #E30613 0%, #C70039 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E30613;
    }
    .stButton>button {
        background-color: #E30613;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #C70039;
    }
</style>
""", unsafe_allow_html=True)

# Load model artifact
@st.cache_resource
def load_model():
    """Load the trained model artifact"""
    model_path = Path(__file__).parent.parent / "models" / "linear_regression_tunisia_cars.pkl"
    
    if not model_path.exists():
        st.error(f"❌ Model file not found at {model_path}")
        st.info("Please run the training notebook first to generate the model.")
        st.stop()
    
    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load dataset for dropdowns
@st.cache_data
def load_dataset_info():
    """Load dataset to populate dropdown options"""
    data_path = Path(__file__).parent.parent / "data" / "raw" / "tunisia_cars_dataset.csv"
    
    if not data_path.exists():
        st.warning("Dataset not found. Using default options.")
        return None
    
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.warning(f"Could not load dataset: {str(e)}")
        return None

def prepare_input(user_input, artifact):
    """Prepare user input for prediction"""
    # Create DataFrame
    df_input = pd.DataFrame([user_input])
    
    # Feature engineering (same as training)
    df_input['age'] = 2025 - df_input['year']
    df_input['mileage_per_year'] = df_input['mileage'] / df_input['age'].replace(0, 1)
    
    # One-hot encode categorical columns
    cat_cols = artifact['categorical_columns']
    df_encoded = pd.get_dummies(df_input, columns=cat_cols, drop_first=True)
    
    # Align columns with training data
    final_cols = artifact['feature_columns']
    for col in final_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[final_cols]
    
    # Scale numeric features
    numeric_cols = artifact['numeric_columns']
    df_encoded[numeric_cols] = artifact['scaler'].transform(df_encoded[numeric_cols])
    
    return df_encoded

def predict_price(user_input, artifact):
    """Make prediction"""
    X_prepared = prepare_input(user_input, artifact)
    prediction = artifact['model'].predict(X_prepared)[0]
    return max(0, prediction)  # Ensure non-negative

# Load model and data
artifact = load_model()
dataset = load_dataset_info()

# Extract unique values from dataset
if dataset is not None:
    brands = sorted(dataset['brand'].unique().tolist())
    models_by_brand = {brand: sorted(dataset[dataset['brand'] == brand]['model'].unique().tolist()) 
                       for brand in brands}
    locations = sorted(dataset['location'].unique().tolist())
    colors = sorted(dataset['color'].unique().tolist())
else:
    # Default values
    brands = ["Peugeot", "Renault", "Volkswagen", "Kia", "Hyundai", "Toyota", "Suzuki", 
              "Dacia", "Fiat", "BMW", "Mercedes", "Audi", "Opel", "Ford", "Nissan", 
              "Mitsubishi", "Chery"]
    models_by_brand = {brand: [f"{brand} Model"] for brand in brands}
    locations = ["Tunis", "Sfax", "Sousse", "Ariana", "Ben Arous", "Nabeul", 
                 "Bizerte", "Gabes", "Kairouan", "Gafsa"]
    colors = ["blanc", "noir", "gris", "argent", "rouge", "bleu", "vert", "beige"]

fuel_types = ["Essence", "Diesel", "Hybride"]
gearbox_types = ["Manuelle", "Automatique"]
conditions = ["excellent", "tres bon etat", "bon etat", "moyen", "a reparer"]
car_bodies = ["citadine", "compacte", "berline", "SUV", "break", "monospace", "pickup"]
import_types = ["local", "imported"]

# Header
st.markdown('<div class="main-header">🚗 Tunisia Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Get instant price estimates for used cars in Tunisia</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    flag_path = Path(__file__).parent.parent / "images" / "Flag_of_Tunisia.png"
    if flag_path.exists():
        st.image(str(flag_path), width=100)
    else:
        st.image("🇹🇳")
    st.title("About")
    st.info(
        """
        This app predicts used car prices in Tunisia using a **Linear Regression** model 
        trained on 60,000+ car listings across all 24 governorates.
        """
    )
    
    st.subheader("📊 Model Performance")
    metrics = artifact.get('metrics', {})
    if metrics:
        st.metric("R² Score", f"{metrics.get('r2_test', 0):.4f}")
        st.metric("MAE", f"{metrics.get('mae', 0):,.0f} TND")
        st.metric("RMSE", f"{metrics.get('rmse', 0):,.0f} TND")
        st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    
    st.markdown("---")
    st.markdown("**Developed by:** Khalil Amamri")
    st.markdown("**Model:** Linear Regression")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔧 Car Details")
    
    # Basic Information
    st.markdown("##### Basic Information")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        selected_brand = st.selectbox("Brand", brands, index=brands.index("Renault") if "Renault" in brands else 0)
    
    with c2:
        available_models = models_by_brand.get(selected_brand, [selected_brand])
        selected_model = st.selectbox("Model", available_models)
    
    with c3:
        year = st.slider("Year", 2005, 2024, 2016)
    
    c4, c5, c6 = st.columns(3)
    
    with c4:
        mileage = st.number_input("Mileage (km)", 0, 500000, 125000, step=5000)
    
    with c5:
        fuel = st.selectbox("Fuel Type", fuel_types)
    
    with c6:
        gearbox = st.selectbox("Gearbox", gearbox_types)
    
    # Technical Specifications
    st.markdown("##### Technical Specifications")
    c7, c8, c9, c10 = st.columns(4)
    
    with c7:
        car_body = st.selectbox("Body Type", car_bodies)
    
    with c8:
        horsepower = st.number_input("Horsepower", 60, 300, 90)
    
    with c9:
        engine_size = st.number_input("Engine Size (L)", 0.9, 4.0, 1.2, step=0.1)
    
    with c10:
        condition = st.selectbox("Condition", conditions, index=1)
    
    # Ownership & History
    st.markdown("##### Ownership & History")
    c11, c12, c13 = st.columns(3)
    
    with c11:
        num_owners = st.number_input("Number of Owners", 1, 5, 2)
    
    with c12:
        accident = st.selectbox("Accident History", ["No", "Yes"])
        accident_history = 1 if accident == "Yes" else 0
    
    with c13:
        import_local = st.selectbox("Origin", import_types)
    
    # Location & Color
    st.markdown("##### Location & Color")
    c14, c15 = st.columns(2)
    
    with c14:
        location = st.selectbox("Location", locations, index=locations.index("Tunis") if "Tunis" in locations else 0)
    
    with c15:
        color = st.selectbox("Color", colors)
    
    # Features
    st.markdown("##### Features & Options")
    c16, c17, c18, c19 = st.columns(4)
    
    with c16:
        air_conditioning = st.checkbox("Air Conditioning", value=True)
        parking_sensor = st.checkbox("Parking Sensor", value=True)
    
    with c17:
        rear_camera = st.checkbox("Rear Camera")
        sunroof = st.checkbox("Sunroof")
    
    with c18:
        alloy_wheels = st.checkbox("Alloy Wheels", value=True)
        bluetooth = st.checkbox("Bluetooth", value=True)
    
    with c19:
        gps = st.checkbox("GPS")

# Prediction column
with col2:
    st.subheader("💰 Price Prediction")
    
    if st.button("🔮 Predict Price", use_container_width=True):
        # Prepare input
        user_input = {
            'brand': selected_brand,
            'model': selected_model,
            'year': year,
            'mileage': mileage,
            'fuel': fuel,
            'gearbox': gearbox,
            'vehicle_condition': condition,
            'car_body': car_body,
            'horsepower': horsepower,
            'engine_size': engine_size,
            'number_of_owners': num_owners,
            'accident_history': accident_history,
            'import_or_local': import_local,
            'location': location,
            'color': color,
            'air_conditioning': int(air_conditioning),
            'parking_sensor': int(parking_sensor),
            'rear_camera': int(rear_camera),
            'sunroof': int(sunroof),
            'alloy_wheels': int(alloy_wheels),
            'bluetooth': int(bluetooth),
            'gps': int(gps)
        }
        
        # Make prediction
        try:
            with st.spinner("Calculating price..."):
                predicted_price = predict_price(user_input, artifact)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <div style="font-size: 1.2rem;">Estimated Price</div>
                    <div class="prediction-price">{predicted_price:,.0f} TND</div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">≈ {predicted_price/3.3:,.0f} USD</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Price range (confidence interval approximation)
                margin = predicted_price * 0.15  # ±15% margin
                st.success(f"**Price Range:** {predicted_price - margin:,.0f} - {predicted_price + margin:,.0f} TND")
                
                # Car summary
                st.markdown("#### 📋 Car Summary")
                st.markdown(f"""
                - **Car:** {selected_brand} {selected_model}
                - **Year:** {year} ({2025 - year} years old)
                - **Mileage:** {mileage:,} km
                - **Condition:** {condition}
                - **Location:** {location}
                """)
                
                # Key factors
                st.markdown("#### 🔑 Key Factors")
                age = 2025 - year
                mileage_per_year = mileage / max(age, 1)
                
                st.metric("Age", f"{age} years", 
                         delta="Newer" if age < 5 else "Older" if age > 10 else "Average",
                         delta_color="inverse")
                st.metric("Annual Mileage", f"{mileage_per_year:,.0f} km/year",
                         delta="Low" if mileage_per_year < 12000 else "High" if mileage_per_year > 20000 else "Average",
                         delta_color="inverse" if mileage_per_year < 12000 else "normal")
                
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check your inputs and try again.")

# Feature Importance Section
st.markdown("---")
st.subheader("📊 Model Insights")

col_insight1, col_insight2 = st.columns(2)

with col_insight1:
    st.markdown("##### Most Important Features")
    st.markdown("""
    The model considers multiple factors:
    - **Age & Mileage**: Older cars with higher mileage typically cost less
    - **Brand & Model**: Premium brands command higher prices
    - **Condition**: Excellent condition significantly increases value
    - **Features**: Modern features like GPS, rear camera add value
    - **Location**: Urban areas may have different pricing
    """)

with col_insight2:
    st.markdown("##### Tips for Better Valuation")
    st.markdown("""
    - 🔧 **Regular maintenance** preserves value
    - 📉 **Avoid accidents** to maintain higher resale value
    - ⭐ **Keep car in excellent condition**
    - 🚗 **Lower annual mileage** increases value
    - 📍 **Consider location** when pricing
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🚗 Tunisia Car Price Predictor | Powered by Machine Learning</p>
    <p>Data based on 60,000+ car listings across all 24 governorates of Tunisia</p>
</div>
""", unsafe_allow_html=True)
