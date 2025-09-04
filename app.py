# File: app.py
import streamlit as st
import pandas as pd
from src.data_processing import fetch_world_bank_data, preprocess_data
from src.model_training import prepare_data, train_model, evaluate_model, make_prediction, MODELS
from src.visualization import plot_indicator_trend, plot_predictions_vs_actuals

# --- Page Configuration ---
st.set_page_config(page_title="Economic Predictor",
                   page_icon="📈", layout="wide")

# --- Constants & Configuration ---
COUNTRIES = {"Brazil": "BRA", "Canada": "CAN"}
INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP (current US$)',
    'SP.POP.TOTL': 'Population, total',
    'EN.ATM.CO2E.KT': 'CO2 emissions (kt)',
    'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
    'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)'
}
START_YEAR = 2000
END_YEAR = 2023
TARGET_COLUMN = 'GDP (current US$)'
FEATURES = list(INDICATORS.values())
# Ensure the target column is not in the features list for training
if TARGET_COLUMN in FEATURES:
    FEATURES.remove(TARGET_COLUMN)
FEATURES.append('Year')


# --- Data Loading ---
@st.cache_data
def load_data():
    """Fetches and preprocesses data from the World Bank. Caches the result."""
    st.write("Fetching and preprocessing data... This may take a moment.")
    raw_data = fetch_world_bank_data(
        list(COUNTRIES.values()), INDICATORS, START_YEAR, END_YEAR)
    processed_data = preprocess_data(raw_data)
    return processed_data


# --- Main Application Logic ---
st.title("📈 Economic Indicator Prediction Dashboard")

# Load data once and cache it
main_data = load_data()

# --- Sidebar for User Inputs ---
st.sidebar.header("⚙️ Configuration")
selected_country_name = st.sidebar.selectbox(
    "Select Country", list(COUNTRIES.keys()))
selected_country_code = COUNTRIES[selected_country_name]

selected_model = st.sidebar.selectbox("Select Model", list(MODELS.keys()))

st.sidebar.markdown("---")
if st.sidebar.button("Train Model and Predict", type="primary"):
    # Filter data for the selected country
    country_data = main_data[main_data['economy']
                             == selected_country_code].copy()

    # 1. Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(
        country_data, TARGET_COLUMN, FEATURES)
    st.session_state['X_test'] = X_test  # Store for later use
    st.session_state['y_test'] = y_test

    # 2. Train Model
    model = train_model(X_train, y_train, selected_model)
    st.session_state['trained_model'] = model

    # 3. Evaluate Model
    metrics = evaluate_model(model, X_test, y_test)
    st.session_state['metrics'] = metrics

    # 4. Make Prediction for the next year
    last_known_features = country_data.sort_values(
        by='Year').iloc[[-1]][FEATURES]
    next_year_prediction = make_prediction(model, last_known_features)
    st.session_state['prediction'] = next_year_prediction


# --- Main Application Body ---

st.header(f"Data and Trends for {selected_country_name}")

# Display preprocessed data for the selected country
st.dataframe(main_data[main_data['economy'] == selected_country_code])

# Display a plot for a selected indicator
indicator_to_plot = st.selectbox(
    "Select indicator to visualize", list(INDICATORS.values()))
if indicator_to_plot:
    country_df_viz = main_data[main_data['economy'] == selected_country_code]
    fig_trend = plot_indicator_trend(
        country_df_viz, indicator_to_plot, f"{indicator_to_plot} for {selected_country_name}")
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

st.header(f"Model Training Results for {selected_country_name}")

if 'metrics' in st.session_state:
    st.subheader("Prediction for Next Year")
    pred_value = st.session_state['prediction']
    st.info(
        f"**Predicted {TARGET_COLUMN} for {END_YEAR + 1}:** `${pred_value:,.2f}`")

    st.subheader("Model Performance Metrics")
    metrics = st.session_state['metrics']
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.2f}")
    col2.metric("Mean Squared Error (MSE)", f"${metrics['MSE']:,.2f}")
    col3.metric("R-squared (R²)", f"{metrics['R2 Score']:.3f}")

    st.subheader("Actual vs. Predicted Values on Test Set")
    model = st.session_state['trained_model']
    y_pred = model.predict(st.session_state['X_test'])
    fig_pred = plot_predictions_vs_actuals(
        st.session_state['y_test'], y_pred, f"Model Predictions vs. Actuals for {selected_country_name}")
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Click 'Train Model and Predict' in the sidebar to see the results.")
