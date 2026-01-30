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

# Chart 1: Price Categories (Simplified)
st.markdown("---")
st.subheader("1️⃣ How Many Cars in Each Price Range?")

col1, col2 = st.columns([3, 1])

with col1:
    # Create price categories
    df['price_category'] = pd.cut(
        df['price'],
        bins=[0, 30000, 60000, 100000, float('inf')],
        labels=['Budget\n(< 30k TND)', 'Mid-Range\n(30k-60k TND)', 'Premium\n(60k-100k TND)', 'Luxury\n(> 100k TND)']
    )
    
    category_counts = df['price_category'].value_counts().sort_index()
    
    fig_category = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Price Category', 'y': 'Number of Cars'},
        title='Most Cars Are in the Budget and Mid-Range Categories',
        text=category_counts.values,
        color=category_counts.values,
        color_continuous_scale='Reds'
    )
    fig_category.update_traces(
        texttemplate='%{text:,}',
        textposition='outside'
    )
    fig_category.update_layout(
        showlegend=False,
        height=500,
        xaxis_title="Price Category",
        yaxis_title="Number of Cars"
    )
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.markdown("""
    ##### 📌 What Does This Mean?
    
    **Most cars** fall in the Budget and Mid-Range categories (under 60,000 TND).
    
    **Why it matters:** If your car's predicted price is in a category with many cars, it's a common market segment.
    """)

# Chart 2: Price vs Year (Depreciation)
st.markdown("---")
st.subheader("2️⃣ Newer Cars = Higher Prices")

col3, col4 = st.columns([3, 1])

with col3:
    # Create scatter plot with manual trend line (no statsmodels needed)
    sample_df = df.sample(min(5000, len(df)), random_state=42)
    
    # Create scatter plot
    fig_year = px.scatter(
        sample_df,
        x='year',
        y='price',
        labels={'year': 'Year of Manufacture', 'price': 'Price (TND)'},
        title='How Car Year Affects Price (showing 5,000 sample cars)',
        opacity=0.3,
        color_discrete_sequence=['#FF6B6B']
    )
    
    # Add manual trend line using numpy
    z = np.polyfit(sample_df['year'], sample_df['price'], 1)
    p = np.poly1d(z)
    years_sorted = np.sort(sample_df['year'].unique())
    
    fig_year.add_trace(go.Scatter(
        x=years_sorted,
        y=p(years_sorted),
        mode='lines',
        name='Average Trend Line',
        line=dict(color='#E30613', width=3, dash='solid')
    ))
    
    fig_year.update_layout(
        height=500,
        showlegend=True,
        xaxis_title="Year of Manufacture",
        yaxis_title="Price (TND)",
        yaxis=dict(tickformat=",.0f")
    )
    st.plotly_chart(fig_year, use_container_width=True)

with col4:
    st.markdown("""
    ##### 📌 What Does This Mean?
    
    **The red line goes up** = Newer cars (right side) cost more than older cars (left side).
    
    **Each dot** = One car listing.
    
    **Why it matters:** Year is the #1 factor affecting price.
    """)

# Chart 3: Price Distribution
st.markdown("---")
st.subheader("3️⃣ Where Do Most Car Prices Fall?")

col5, col6 = st.columns([3, 1])

with col5:
    # Calculate key statistics
    median_price = df['price'].median()
    mean_price = df['price'].mean()
    
    fig_dist = px.histogram(
        df,
        x='price',
        nbins=50,
        labels={'price': 'Price (TND)', 'count': 'Number of Cars'},
        title='Most Cars Are Priced Between 20,000 - 60,000 TND',
        color_discrete_sequence=['#E30613']
    )
    
    # Add vertical line for median
    fig_dist.add_vline(
        x=median_price,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_price:,.0f} TND",
        annotation_position="top"
    )
    
    fig_dist.update_layout(
        height=500,
        showlegend=False,
        xaxis_title="Price (TND)",
        yaxis_title="Number of Cars",
        xaxis=dict(tickformat=",.0f")
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col6:
    st.markdown(f"""
    ##### 📌 What Does This Mean?
    
    **The tall bars** show where most cars are priced.
    
    **Green line** = Half of cars cost less, half cost more ({median_price:,.0f} TND)
    
    **Why it matters:** If your car's price is near the green line, it's typical for the market.
    """)

# Chart 4: Top Features Impact (Feature Importance)
st.markdown("---")
st.subheader("4️⃣ What Affects Car Price the Most?")

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
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(10)
        
        # Better feature name mapping
        name_map = {
            'year': '📅 Year (Newer = Higher)',
            'mileage': '🛣️ Mileage (Lower = Higher)',
            'age': '⏰ Age (Younger = Higher)',
            'engine_size': '⚙️ Engine Size',
            'doors': '🚪 Number of Doors',
            'mileage_per_year': '📊 Yearly Mileage',
            'horsepower': '💪 Horsepower',
            'fuel_consumption': '⛽ Fuel Consumption'
        }
        
        feature_importance['Feature_Clean'] = feature_importance['Feature'].apply(
            lambda x: name_map.get(x, x.replace('_', ' ').title())
        )
        
        # Sort by coefficient value (positive to negative)
        feature_importance = feature_importance.sort_values('Coefficient', ascending=True)
        
        fig_importance = px.bar(
            feature_importance,
            x='Coefficient',
            y='Feature_Clean',
            orientation='h',
            labels={'Coefficient': 'Impact on Price (TND)', 'Feature_Clean': 'Feature'},
            title='Top 10 Features That Increase (→) or Decrease (←) Price',
            color='Coefficient',
            color_continuous_scale='RdBu_r'
        )
        fig_importance.update_layout(
            height=500,
            showlegend=False,
            xaxis_title="Impact on Price (TND)",
            yaxis_title="",
            xaxis=dict(tickformat=",.0f")
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("Model artifact not found. Train the model to see feature importance.")

with col8:
    st.markdown("""
    ##### 📌 What Does This Mean?
    
    **Bars going right (red)** = Features that **increase** price
    
    **Bars going left (blue)** = Features that **decrease** price
    
    **Longer bars** = Bigger impact
    
    **Why it matters:** Focus on the top features when evaluating a car's value.
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
