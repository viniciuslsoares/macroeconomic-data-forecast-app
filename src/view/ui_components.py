import streamlit as st
from datetime import datetime
import pandas as pd

def render_sidebar(countries: dict, model_names: list, indicators: dict) -> dict:
    """Renders the sidebar and returns selected configuration."""
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        st.title("⚙️ Configuration Panel")
        
        country_names = list(countries.keys())
        selected_country_name = st.selectbox("Select Country", country_names)
        
        selected_model = st.selectbox("Select Model", model_names)
        selected_target = st.selectbox("Select Target Column", list(indicators.values()))
        
        st.markdown("---")
        
        train_clicked = st.button("Train Model & Predict", type="primary", use_container_width=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Or upload a previously trained model", type=["pkl"])
        
        return {
            "country_name": selected_country_name,
            "country_code": countries[selected_country_name],
            "model": selected_model,
            "target": selected_target,
            "train_clicked": train_clicked,
            "uploaded_file": uploaded_file
        }

def render_metrics_section(metrics: dict, is_economic: bool):
    """Renders the metrics cards."""
    col1, col2 = st.columns(2)
    
    fmt = "${:,.0f}" if is_economic else "{:,.0f}"
    
    with col1:
        with st.container(border=True):
            st.subheader("📊 Model Performance Metrics")
            st.metric("Mean Absolute Error (MAE)", fmt.format(metrics['mae']))
            st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}")
            
    with col2:
        with st.container(border=True):
            st.subheader("🤖 Model Info")
            st.info("Metrics calculated on unseen test data.")

def render_prediction_banner(prediction: float, target_name: str, year: int, is_economic: bool):
    """Renders the big prediction number."""
    with st.container(border=True):
        st.subheader(f"🔮 Prediction for {target_name} in {year}")
        
        fmt = "${:,.0f}" if is_economic else "{:,.0f}"
        st.metric(label="Predicted Value", value=fmt.format(prediction))

def apply_custom_styles():
    """Injects CSS."""
    st.markdown("""
        <style>
            .stMetric { user-select: none; }
            /* Add your other CSS here */
        </style>
    """, unsafe_allow_html=True)