import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.model.model_registry import get_model_names
from src.view.visualization import plot_indicator_trend, plot_predictions_vs_actuals, prepare_plot_data
from src.model.model_training import run_training_pipeline
from src.model.data_processing import fetch_world_bank_data
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Economic Indicator Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for improved UX - make static text non-selectable
st.markdown("""
    <style>
        /* Make static text elements non-selectable and use default cursor */
        p, h1, h2, h3, h4, h5, h6, div, span, label, 
        .stMarkdown, .stText, .stMetric, .stCaption,
        .element-container, .stAlert, .stInfo, .stWarning, .stError, .stSuccess {
            user-select: none;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            cursor: default;
        }
        
        /* Exceptions: Keep inputs, textareas, and code blocks selectable */
        input, textarea, code, pre, .stCodeBlock {
            user-select: text !important;
            -webkit-user-select: text !important;
            -moz-user-select: text !important;
            -ms-user-select: text !important;
            cursor: text;
        }
        
        /* Links should have pointer cursor */
        a {
            cursor: pointer !important;
            user-select: none;
        }
        
        /* Selectboxes, buttons, and other interactive elements should have pointer cursor */
        button, select, [role="button"], [role="option"],
        .stSelectbox, .stButton, .stSlider, .stCheckbox {
            cursor: pointer !important;
        }
        
        /* DataFrames and tables should allow text selection for copying */
        table, .stDataFrame, .dataframe {
            user-select: text !important;
            cursor: default;
        }
        
        /* Plotly charts should have default cursor */
        .js-plotly-plot, .plotly {
            cursor: default !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for model history
if 'model_history' not in st.session_state:
    st.session_state['model_history'] = {}

COUNTRIES = {"Brazil": "BRA", "Canada": "CAN"}
INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP (current US$)',
    'SP.POP.TOTL': 'Population, total',
    'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
    'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)'
}
START_YEAR = 2000
END_YEAR = 2018


@st.cache_data
def load_data():
    """Fetches and preprocesses data from the World Bank. Caches the result."""
    with st.spinner("Fetching and preprocessing data from the World Bank API... This may take a moment."):
        data = fetch_world_bank_data(
            list(COUNTRIES.values()), START_YEAR, END_YEAR, INDICATORS)
    return data


main_data = load_data()

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.title("⚙️ Configuration Panel")
    st.markdown(
        "Select the country and the machine learning model to be used for the prediction.")

    country_names = list(COUNTRIES.keys())
    selected_country_name = st.selectbox(
        "Select Country", country_names)
    selected_country_index = country_names.index(selected_country_name)

    selected_model = st.selectbox("Select Model", get_model_names())

    selected_target_column = st.selectbox(
        "Select Target Column", list(INDICATORS.values()))

    st.markdown("---")

    # The button click triggers the training and saves results to session_state
    if st.button("Train Model & Predict", type="primary", use_container_width=True):
        with st.spinner("Training model... Please wait."):

            # 1. Execute the pipeline
            # END_YEAR is defined globally in app.py
            country_data = main_data[selected_country_index].copy()

            results = run_training_pipeline(
                country_data=country_data,
                target_column=selected_target_column,
                model_name=selected_model,
                end_year=END_YEAR
            )

            # 2. Save results to session (for backward compatibility with existing tabs)
            # This replaces all individual 'st.session_state[...]' = ... assignments
            for key, value in results.items():
                st.session_state[key] = value

            # Save the selected country name for the charts
            st.session_state['selected_country_name'] = selected_country_name

            # 3. Add to model history with unique identifier
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_id = f"{selected_model} - {selected_target_column} - {timestamp}"

            # Create a copy of results with additional metadata
            history_entry = results.copy()
            history_entry['model_name'] = selected_model
            history_entry['target_column'] = selected_target_column
            history_entry['country_name'] = selected_country_name
            history_entry['timestamp'] = timestamp

            st.session_state['model_history'][model_id] = history_entry

            st.success("Model trained successfully!")

st.title("📈 Economic Indicator Prediction Dashboard")
st.markdown(f"Currently analyzing: **{selected_country_name}**")

tab1, tab2, tab3 = st.tabs(["📊 Data Exploration & Visualization",
                            "🧠 Model Training & Prediction",
                            "⚖️ Model Comparison"])

with tab1:
    st.header("Initial Data Analysis", divider='rainbow')

    with st.container(border=True):
        st.subheader("Preprocessed Data Table")
        st.caption(
            f"Displaying the last 10 years of available data for {selected_country_name}. The model will be trained on this dataset.")
        st.dataframe(main_data[selected_country_index].tail(10))

    st.write("")

    with st.container(border=True):
        st.subheader("Indicator Trends Over Time")
        indicator_to_plot = st.selectbox(
            "Select an indicator to visualize its trend:",
            list(INDICATORS.values()),
            key="indicator_selector"
        )
        if indicator_to_plot:
            country_df_viz = main_data[selected_country_index]
            fig_trend = plot_indicator_trend(
                country_df_viz, indicator_to_plot, f"{indicator_to_plot} for {selected_country_name}")
            st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("Machine Learning Results", divider='rainbow')

    if 'metrics' not in st.session_state:
        st.info(
            "Please click the 'Train Model & Predict' button in the sidebar to see the results.")
        st.image(
            "https://media1.tenor.com/m/y2uA8hd_3tEAAAAC/what-are-you-waiting-for-do-it.gif", width=300)
    else:
        ECONOMIC_INDICATORS = {'GDP (current US$)'}
        is_economic = st.session_state['target_column'] in ECONOMIC_INDICATORS

        with st.container(border=True):
            st.subheader(
                f"🔮 Prediction for {st.session_state['target_column']} in {END_YEAR + 1}")
            pred_value = st.session_state['prediction']

            if is_economic:
                formatted_pred = f"${pred_value:,.0f}"
            else:
                formatted_pred = f"{pred_value:,.0f}"

            st.metric(
                label=f"Predicted {st.session_state['target_column']}",
                value=formatted_pred,
                help="This is the model's prediction for the next year based on the latest available data."
            )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("📊 Model Performance Metrics")
                st.caption(
                    "These metrics evaluate the model's accuracy on the unseen test dataset.")
                metrics = st.session_state['metrics']

                if is_economic:
                    formatted_mae = f"${metrics['mae']:,.0f}"
                else:
                    formatted_mae = f"{metrics['mae']:,.0f}"

                st.metric("Mean Absolute Error (MAE)",
                          formatted_mae)
                st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}")

        with col2:
            with st.container(border=True):
                st.subheader("🤖 Model Used")
                st.caption(
                    "This was the algorithm selected for the training process.")
                st.info(f"**Model:** {selected_model}")
                if "Forest" in selected_model or "Boosting" in selected_model:
                    st.markdown(
                        "This is a tree-based ensemble model, often powerful for tabular data.")
                else:
                    st.markdown(
                        "This is a linear model, great for finding simple relationships in data.")

        st.write("")

        with st.container(border=True):
            st.subheader("🎯 Actual vs. Predicted Values (on Test Set)")
            st.caption("This chart helps to visually assess the model's performance by comparing its predictions against the actual historical data it did not see during training, and includes a 5-year forecast.")

            # Prepare plot data using the View layer function
            model_entry = {
                'trained_model': st.session_state['trained_model'],
                'X_train': st.session_state['X_train'],
                'y_train': st.session_state['y_train'],
                'X_test': st.session_state['X_test'],
                'y_test': st.session_state['y_test']
            }

            combined_X, combined_y_actual, combined_y_pred = prepare_plot_data(
                model_entry, END_YEAR
            )

            fig_pred = plot_predictions_vs_actuals(
                combined_X,
                combined_y_actual,
                combined_y_pred,
                f"Model Predictions vs. Actuals for {selected_country_name} (with 5-year forecast)",
                st.session_state['target_column']
            )
            st.plotly_chart(fig_pred, use_container_width=True)

with tab3:
    st.header("Model Comparison", divider='rainbow')

    model_history = st.session_state.get('model_history', {})

    if len(model_history) < 2:
        st.warning(
            "⚠️ Train at least two models to compare. Please train models using the 'Model Training & Prediction' tab.")
        st.info("💡 Tip: Train different models or the same model with different configurations to compare their performance.")
    else:
        # Get list of model identifiers for selection
        model_ids = list(model_history.keys())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model A (Reference)")
            model_a_id = st.selectbox(
                "Select Model A:",
                model_ids,
                key="model_a_selector",
                help="This model will be used as the reference for comparison."
            )

        with col2:
            st.subheader("Model B (Challenger)")
            # Filter out model_a_id from options for model_b
            model_b_options = [mid for mid in model_ids if mid != model_a_id]
            if not model_b_options:
                st.warning(
                    "Only one model available. Train another model to compare.")
                model_b_id = None
            else:
                model_b_id = st.selectbox(
                    "Select Model B:",
                    model_b_options,
                    key="model_b_selector",
                    help="This model will be compared against Model A."
                )

        if model_b_id:
            model_a = model_history[model_a_id]
            model_b = model_history[model_b_id]

            st.write("")

            # Display model information
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                with st.container(border=True):
                    st.markdown(f"**Model:** {model_a['model_name']}")
                    st.markdown(f"**Target:** {model_a['target_column']}")
                    st.markdown(f"**Country:** {model_a['country_name']}")
                    st.caption(f"Trained: {model_a['timestamp']}")

            with info_col2:
                with st.container(border=True):
                    st.markdown(f"**Model:** {model_b['model_name']}")
                    st.markdown(f"**Target:** {model_b['target_column']}")
                    st.markdown(f"**Country:** {model_b['country_name']}")
                    st.caption(f"Trained: {model_b['timestamp']}")

            st.write("")

            # Metrics comparison
            st.subheader("📊 Performance Metrics Comparison")

            metrics_a = model_a['metrics']
            metrics_b = model_b['metrics']

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            ECONOMIC_INDICATORS = {'GDP (current US$)'}
            is_economic = model_a['target_column'] in ECONOMIC_INDICATORS

            with metric_col1:
                st.metric(
                    label="Mean Absolute Error (MAE)",
                    value=f"${metrics_b['mae']:,.0f}" if is_economic else f"{metrics_b['mae']:,.0f}",
                    delta=f"{metrics_b['mae'] - metrics_a['mae']:,.0f}",
                    delta_color="inverse",
                    help="Lower is better. Delta shows Model B - Model A."
                )

            with metric_col2:
                st.metric(
                    label="R-squared (R²)",
                    value=f"{metrics_b['r2_score']:.3f}",
                    delta=f"{metrics_b['r2_score'] - metrics_a['r2_score']:.3f}",
                    delta_color="normal",
                    help="Higher is better. Delta shows Model B - Model A."
                )

            with metric_col3:
                st.metric(
                    label="Mean Squared Error (MSE)",
                    value=f"${metrics_b['mse']:,.0f}" if is_economic else f"{metrics_b['mse']:,.0f}",
                    delta=f"{metrics_b['mse'] - metrics_a['mse']:,.0f}",
                    delta_color="inverse",
                    help="Lower is better. Delta shows Model B - Model A."
                )

            st.write("")

            # Predictions comparison
            st.subheader("🎯 Predictions Comparison")

            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                with st.container(border=True):
                    st.markdown(f"**Model A Prediction**")
                    pred_a = model_a['prediction']
                    formatted_pred_a = f"${pred_a:,.0f}" if is_economic else f"{pred_a:,.0f}"
                    st.metric(
                        label=f"Predicted {model_a['target_column']}",
                        value=formatted_pred_a
                    )

            with pred_col2:
                with st.container(border=True):
                    st.markdown(f"**Model B Prediction**")
                    pred_b = model_b['prediction']
                    formatted_pred_b = f"${pred_b:,.0f}" if is_economic else f"{pred_b:,.0f}"
                    st.metric(
                        label=f"Predicted {model_b['target_column']}",
                        value=formatted_pred_b,
                        delta=f"{pred_b - pred_a:,.0f}" if is_economic else f"{pred_b - pred_a:,.0f}",
                        help="Difference: Model B - Model A"
                    )

            st.write("")

            # Visualizations comparison
            st.subheader("📈 Predictions vs Actuals Comparison")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                st.markdown(f"**Model A: {model_a['model_name']}**")
                # Prepare data for Model A using the View layer function
                combined_X_a, combined_y_actual_a, combined_y_pred_a = prepare_plot_data(
                    model_a, END_YEAR
                )

                fig_a = plot_predictions_vs_actuals(
                    combined_X_a,
                    combined_y_actual_a,
                    combined_y_pred_a,
                    f"Model A: {model_a['model_name']}",
                    model_a['target_column']
                )
                st.plotly_chart(fig_a, use_container_width=True)

            with viz_col2:
                st.markdown(f"**Model B: {model_b['model_name']}**")
                # Prepare data for Model B using the View layer function
                combined_X_b, combined_y_actual_b, combined_y_pred_b = prepare_plot_data(
                    model_b, END_YEAR
                )

                fig_b = plot_predictions_vs_actuals(
                    combined_X_b,
                    combined_y_actual_b,
                    combined_y_pred_b,
                    f"Model B: {model_b['model_name']}",
                    model_b['target_column']
                )
                st.plotly_chart(fig_b, use_container_width=True)
