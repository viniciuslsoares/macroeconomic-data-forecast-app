import streamlit as st
from datetime import datetime
import pandas as pd

def render_sidebar(countries: dict, model_names: list, indicators: dict) -> dict:
    """
    Renders the sidebar and returns selected configuration.
    Uses st.form to batch inputs and prevent unnecessary reloads (resets).
    """
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
        st.title("⚙️ Configuration Panel")
        
        with st.form("model_configuration_form"):
            
            country_names = list(countries.keys())
            # index=0 (std value)
            selected_country_name = st.selectbox("Select Country", country_names, index=0)
            
            selected_model = st.selectbox("Select Model", model_names, index=0)
            selected_target = st.selectbox("Select Target Column", list(indicators.values()), index=0)
            
            st.markdown("---")
            
            # The page only reloads with this button
            train_clicked = st.form_submit_button("Train Model & Predict", type="primary", width='stretch')
        
        st.markdown("---")
        st.caption("Load Existing Model")
        
        uploaded_file = st.file_uploader("Upload .pkl file", type=["pkl"])
        
        return {
            "country_name": selected_country_name,
            "country_code": countries[selected_country_name],
            "model": selected_model,
            "target": selected_target,
            "train_clicked": train_clicked,
            "uploaded_file": uploaded_file
        }

def render_metrics_section(metrics: dict, model_details: dict, is_economic: bool):
    """
    Renders the metrics cards and model details.
    
    Args:
        metrics: Dictionary containing MAE, MSE, R2.
        model_details: Dictionary containing 'model', 'country', 'target'.
        is_economic: Boolean to format currency.
    """
    col1, col2 = st.columns(2)
    
    fmt = "${:,.0f}" if is_economic else "{:,.0f}"
    
    with col1:
        with st.container(border=True):
            st.subheader("📊 Model Performance")
            st.metric("Mean Absolute Error (MAE)", fmt.format(metrics['mae']))
            st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}")
            
    with col2:
        with st.container(border=True):
            st.subheader("🤖 Model Info")
            st.markdown(f"**Algorithm:** {model_details.get('model', 'Unknown')}")
            st.markdown(f"**Country:** {model_details.get('country', 'Unknown')}")
            st.markdown(f"**Target:** {model_details.get('target', 'Unknown')}")
            st.caption("Metrics calculated on unseen test data (Test Set).")

def render_prediction_banner(prediction: float, target_name: str, year: int, is_economic: bool):
    """Renders the big prediction number."""
    with st.container(border=True):
        st.subheader(f"🔮 Prediction for {target_name} in {year}")
        
        fmt = "${:,.0f}" if is_economic else "{:,.0f}"
        st.metric(label="Predicted Value", value=fmt.format(prediction))

def apply_custom_styles():
    """
    Injects custom CSS to improve the app's visual layout.
    """
    st.markdown("""
        <style>
            /* Prevents text selection on metric widgets for a cleaner UI feel */
            .stMetric { user-select: none; }

            /* Adds top padding to the sidebar to prevent content from touching the edge */
            div[data-testid="stSidebarUserContent"] {
                padding-top: 2rem;
            }
        </style>
    """, unsafe_allow_html=True) # Required argument to allow raw HTML/CSS