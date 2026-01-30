import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Market Insights - Tunisia Cars",
    page_icon="📊",
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
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #E30613;
        margin-bottom: 1rem;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    """Load the car dataset"""
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "tunisia_cars_dataset.csv"
    
    if not data_path.exists():
        st.error("❌ Dataset not found. Please generate the dataset first.")
        st.stop()
    
    try:
        df = pd.read_csv(data_path)
        # Add calculated features
        df['age'] = 2025 - df['year']
        df['mileage_per_year'] = df['mileage'] / df['age'].replace(0, 1)
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.stop()

# Load model
@st.cache_resource
def load_model():
    """Load trained model for feature importance"""
    model_path = Path(__file__).parent.parent.parent / "models" / "linear_regression_tunisia_cars.pkl"
    
    if not model_path.exists():
        return None
    
    try:
        artifact = joblib.load(model_path)
        return artifact
    except:
        return None

# Load data
df = load_data()
artifact = load_model()

# Sidebar
with st.sidebar:
    st.title("📊 Market Insights")
    st.markdown("---")
    st.info("Explore the Tunisian car market through interactive visualizations.")
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Overview")
    st.metric("Total Listings", f"{len(df):,}")
    st.metric("Brands", df['brand'].nunique())
    st.metric("Avg Price", f"{df['price'].mean():,.0f} TND")

# Header
st.markdown('<div class="main-header">📊 Market Insights Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Understanding the Tunisian Car Market</div>', unsafe_allow_html=True)

st.markdown("""
<div class="insight-card">
    <strong>💡 Purpose:</strong> Market overview to help you understand pricing patterns and validate predictions.
</div>
""", unsafe_allow_html=True)

# Chart 1: Average Price by Brand
st.markdown("---")
st.subheader("1️⃣ Average Price by Brand")

col1, col2 = st.columns([3, 1])

with col1:
    # Calculate average price by brand
    brand_avg = df.groupby('brand')['price'].mean().sort_values(ascending=True).tail(15)
    
    fig_brand = px.bar(
        x=brand_avg.values,
        y=brand_avg.index,
        orientation='h',
        labels={'x': 'Average Price (TND)', 'y': 'Brand'},
        title='Top 15 Brands by Average Price',
        color=brand_avg.values,
        color_continuous_scale='Reds'
    )
    fig_brand.update_layout(
        showlegend=False,
        height=500,
        xaxis_title="Average Price (TND)",
        yaxis_title="Brand"
    )
    st.plotly_chart(fig_brand, use_container_width=True)

with col2:
    st.markdown("""
    ##### 📌 Insight
    
    Premium brands (Mercedes, BMW, Audi) have higher average prices while economy brands (Suzuki, Dacia) are more affordable.
    
    **Use this to:** Compare your car's brand against market averages.
    """)

# Chart 2: Price vs Year (Depreciation)
st.markdown("---")
st.subheader("2️⃣ Price vs Year (Car Depreciation)")

col3, col4 = st.columns([3, 1])

with col3:
    # Create scatter plot with manual trend line (no statsmodels needed)
    sample_df = df.sample(min(5000, len(df)))
    
    # Create scatter plot
    fig_year = px.scatter(
        sample_df,
        x='year',
        y='price',
        labels={'year': 'Year of Manufacture', 'price': 'Price (TND)'},
        title='Car Price Decreases with Age',
        opacity=0.4,
        color_discrete_sequence=['#E30613']
    )
    
    # Add manual trend line using numpy
    z = np.polyfit(sample_df['year'], sample_df['price'], 1)
    p = np.poly1d(z)
    years_sorted = np.sort(sample_df['year'].unique())
    
    fig_year.add_trace(go.Scatter(
        x=years_sorted,
        y=p(years_sorted),
        mode='lines',
        name='Trend',
        line=dict(color='blue', width=2)
    ))
    
    fig_year.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_year, use_container_width=True)

with col4:
    st.markdown("""
    ##### 📌 Insight
    
    Cars lose value over time (depreciation). Newer cars cost significantly more than older ones.
    
    **Key Takeaway:** Age is the most important factor in car pricing.
    """)

# Chart 3: Price Distribution
st.markdown("---")
st.subheader("3️⃣ Price Distribution (Market Overview)")

col5, col6 = st.columns([3, 1])

with col5:
    fig_dist = px.histogram(
        df,
        x='price',
        nbins=50,
        labels={'price': 'Price (TND)', 'count': 'Number of Cars'},
        title='Most Cars Are Priced Between 20,000 - 60,000 TND',
        color_discrete_sequence=['#E30613']
    )
    fig_dist.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Price (TND)",
        yaxis_title="Number of Cars"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col6:
    st.markdown(f"""
    ##### 📌 Insight
    
    Most cars are priced between 20,000 - 60,000 TND. Median: **{df['price'].median():,.0f} TND**
    
    **Use this to:** Check if your predicted price falls within a realistic range.
    """)

# Chart 4: Top Features Impact (Feature Importance)
st.markdown("---")
st.subheader("4️⃣ Top Features Impact on Price")

col7, col8 = st.columns([3, 1])

with col7:
    if artifact is not None:
        # Get feature coefficients from model
        model = artifact['model']
        feature_names = artifact['feature_columns']
        coefficients = model.coef_
        
        # Create dataframe of feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        })
        
        # Get absolute values and sort
        feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(15)
        
        # Clean feature names for display
        feature_importance['Feature_Clean'] = feature_importance['Feature'].str.replace('_', ' ').str.title()
        
        fig_importance = px.bar(
            feature_importance,
            x='Coefficient',
            y='Feature_Clean',
            orientation='h',
            labels={'Coefficient': 'Impact on Price', 'Feature_Clean': 'Feature'},
            title='Top 15 Features Affecting Car Price',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig_importance.update_layout(
            height=500,
            showlegend=False,
            xaxis_title="Impact on Price (TND)",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Model artifact not found. Train the model to see feature importance.")

with col8:
    st.markdown("""
    ##### 📌 Insight
    
    Shows which features have the biggest impact on price.
    
    **Most Important:** Age, mileage, brand, engine size, condition
    
    **Least Important:** Color, some optional features
    """)

# Additional insights
st.markdown("---")
st.subheader("🔍 Additional Market Insights")

col9, col10, col11 = st.columns(3)

with col9:
    st.markdown("##### 🚗 Most Popular Brands")
    top_brands = df['brand'].value_counts().head(5)
    for brand, count in top_brands.items():
        st.markdown(f"- **{brand}**: {count:,} listings ({count/len(df)*100:.1f}%)")

with col10:
    st.markdown("##### ⚡ Fuel Type Distribution")
    fuel_dist = df['fuel'].value_counts()
    for fuel, count in fuel_dist.items():
        st.markdown(f"- **{fuel}**: {count/len(df)*100:.1f}%")

with col11:
    st.markdown("##### 📍 Top Locations")
    top_locations = df['location'].value_counts().head(5)
    for loc, count in top_locations.items():
        st.markdown(f"- **{loc}**: {count:,} listings")

# Summary
st.markdown("---")
st.info("""
**💡 Summary:** This dashboard helps you understand market prices, see depreciation patterns, 
validate predictions, and identify key value factors.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>📊 Market Insights Dashboard | Based on 60,000+ car listings</p>
    <p>Data is synthetic and for educational purposes</p>
</div>
""", unsafe_allow_html=True)
