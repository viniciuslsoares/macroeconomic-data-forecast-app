import sys
import os
import io
import joblib

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.model.model_registry import get_model_names
from src.view.visualization import (
    plot_indicator_trend,
    plot_predictions_vs_actuals,
    prepare_plot_data,
    plot_feature_importance
)
from src.model.model_training import (
    run_training_pipeline,
    save_model,
    evaluate_loaded_model,
    make_recursive_future_prediction,
)
from src.model.data_processing import fetch_world_bank_data
from src.view.export_utils import (
    export_train_test_data,
    export_actual_vs_predicted_data,
    export_future_predictions,
)
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Economic Indicator Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for improved UX - make static text non-selectable
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Initialize session state for model history
if "model_history" not in st.session_state:
    st.session_state["model_history"] = {}

COUNTRIES = {"Brazil": "BRA", "Canada": "CAN"}
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SP.POP.TOTL": "Population, total",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
}
START_YEAR = 2000
END_YEAR = 2018


@st.cache_data(show_spinner=False)
def load_data():
    """Fetches and preprocesses data from the World Bank. Caches the result."""
    with st.spinner(
        "⏳ Fetching and preprocessing data from the World Bank API... This may take a moment."
    ):
        data = fetch_world_bank_data(
            list(COUNTRIES.values()), START_YEAR, END_YEAR, INDICATORS
        )
    return data


main_data = load_data()

with st.sidebar:
    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png",
        width=200,
    )
    st.title("⚙️ Configuration Panel")
    st.markdown(
        "Select the country and the machine learning model to be used for the prediction."
    )

    country_names = list(COUNTRIES.keys())
    selected_country_name = st.selectbox("Select Country", country_names)
    selected_country_index = country_names.index(selected_country_name)

    selected_model = st.selectbox("Select Model", get_model_names())

    selected_target_column = st.selectbox(
        "Select Target Column", list(INDICATORS.values())
    )

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
                end_year=END_YEAR,
            )

            # 2. Save results to session (for backward compatibility with existing tabs)
            # This replaces all individual 'st.session_state[...]' = ... assignments
            for key, value in results.items():
                st.session_state[key] = value

            # Save the selected country name for the charts
            st.session_state["selected_country_name"] = selected_country_name

            # 3. Add to model history with unique identifier
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_id = f"{selected_model} - {selected_target_column} - {timestamp}"

            # Create a copy of results with additional metadata
            history_entry = results.copy()
            history_entry["model_name"] = selected_model
            history_entry["target_column"] = selected_target_column
            history_entry["country_name"] = selected_country_name
            history_entry["timestamp"] = timestamp

            st.session_state["model_history"][model_id] = history_entry

            st.success("Model trained successfully!")

    # Load model button
    uploaded_model = st.file_uploader(
        "Or upload a previously trained model", type=["pkl"]
    )
    if uploaded_model is not None:
        try:
            # Load the model from uploaded file bytes
            model = joblib.load(io.BytesIO(uploaded_model.getvalue()))

            # Evaluate the loaded model against existing data to generate all required artifacts
            country_data = main_data[selected_country_index].copy()
            results = evaluate_loaded_model(
                model=model,
                country_data=country_data,
                target_column=selected_target_column,
                end_year=END_YEAR,
            )

            # Save results to session (for full dashboard functionality)
            for key, value in results.items():
                st.session_state[key] = value

            # Save the selected country name for the charts
            st.session_state["selected_country_name"] = selected_country_name

            # Set a flag to indicate this is a loaded model
            st.session_state["model_loaded"] = True

            # Add to model history with unique identifier
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_id = f"Loaded Model - {selected_target_column} - {timestamp}"

            # Create a copy of results with additional metadata
            history_entry = results.copy()
            history_entry["model_name"] = "Loaded Model"
            history_entry["target_column"] = selected_target_column
            history_entry["country_name"] = selected_country_name
            history_entry["timestamp"] = timestamp

            st.session_state["model_history"][model_id] = history_entry

            st.success("Model loaded successfully with full functionality!")

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

st.title("📈 Economic Indicator Prediction Dashboard")
st.markdown(f"Currently analyzing: **{selected_country_name}**")

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Data Exploration & Visualization",
        "🧠 Model Training & Prediction",
        "⚖️ Model Comparison",
    ]
)

with tab1:
    st.header("Initial Data Analysis", divider="rainbow")

    with st.container(border=True):
        st.subheader("Preprocessed Data Table")
        st.caption(
            f"Displaying the last 10 years of available data for {selected_country_name}. The model will be trained on this dataset."
        )

        # Display the data table
        data_table = main_data[selected_country_index].tail(10)
        st.dataframe(data_table)

        # Create metadata for export
        metadata = {
            "model_name": "Preprocessed Data Export",
            "target_column": "All Indicators",
            "country_name": selected_country_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Convert DataFrame to CSV
        csv_buffer = io.BytesIO()
        data_table.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        st.download_button(
            label="Export Preprocessed Data as CSV",
            data=csv_buffer,
            file_name=f"preprocessed_data_{selected_country_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="export_preprocessed_data",
        )

    st.write("")

    with st.container(border=True):
        st.subheader("Indicator Trends Over Time")
        indicator_to_plot = st.selectbox(
            "Select an indicator to visualize its trend:",
            list(INDICATORS.values()),
            key="indicator_selector",
        )
        if indicator_to_plot:
            country_df_viz = main_data[selected_country_index]
            fig_trend = plot_indicator_trend(
                country_df_viz,
                indicator_to_plot,
                f"{indicator_to_plot} for {selected_country_name}",
            )
            st.plotly_chart(fig_trend, use_container_width=True)

with tab2:
    st.header("Machine Learning Results", divider="rainbow")

    if "metrics" not in st.session_state:
        st.info(
            "Please click the 'Train Model & Predict' button in the sidebar to see the results."
        )
    else:
        ECONOMIC_INDICATORS = {"GDP (current US$)"}
        is_economic = st.session_state["target_column"] in ECONOMIC_INDICATORS

        with st.container(border=True):
            st.subheader(
                f"🔮 Prediction for {st.session_state['target_column']} in {END_YEAR + 1}"
            )
            pred_value = st.session_state["prediction"]

            if is_economic:
                formatted_pred = f"${pred_value:,.0f}"
            else:
                formatted_pred = f"{pred_value:,.0f}"

            st.metric(
                label=f"Predicted {st.session_state['target_column']}",
                value=formatted_pred,
                help="This is the model's prediction for the next year based on the latest available data.",
            )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.subheader("📊 Model Performance Metrics")
                st.caption(
                    "These metrics evaluate the model's accuracy on the unseen test dataset."
                )
                metrics = st.session_state["metrics"]

                if is_economic:
                    formatted_mae = f"${metrics['mae']:,.0f}"
                else:
                    formatted_mae = f"{metrics['mae']:,.0f}"

                st.metric("Mean Absolute Error (MAE)", formatted_mae)
                st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}")

        with col2:
            with st.container(border=True):
                st.subheader("🤖 Model Used")
                st.caption("This was the algorithm selected for the training process.")
                st.info(f"**Model:** {selected_model}")
                if "Forest" in selected_model or "Boosting" in selected_model:
                    st.markdown(
                        "This is a tree-based ensemble model, often powerful for tabular data."
                    )
                else:
                    st.markdown(
                        "This is a linear model, great for finding simple relationships in data."
                    )

        st.write("")

        with st.container(border=True):
            st.subheader("🎯 Actual vs. Predicted Values (on Test Set)")
            st.caption(
                "This chart helps to visually assess the model's performance by comparing its predictions against the actual historical data it did not see during training, and includes a 5-year forecast."
            )

            model = st.session_state["trained_model"]
            X_train = st.session_state["X_train"]
            y_train = st.session_state["y_train"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]
            
            last_features = X_test.iloc[-1]
            
            X_history = pd.concat([X_train, X_test])

            if 'lag_1' in last_features:
                future_df = make_recursive_future_prediction(
                    model, 
                    last_features,
                    X_history, 
                    years_ahead=5
                )
                
                y_pred_future = future_df['predicted_value'].values
                future_years = future_df[['year']]
                    
            else:
                st.warning("⚠️ Modelo antigo detectado (sem feature de Lag). A previsão futura será constante.")
                future_years = pd.DataFrame({'year': range(END_YEAR + 1, END_YEAR + 6)})
                other_features = X_test.drop(columns=['year'], errors='ignore').columns
                for feature in other_features:
                    future_years[feature] = last_features[feature]
                y_pred_future = model.predict(future_years)

            y_pred_train = st.session_state["y_pred_train_abs"]
            y_pred_test = st.session_state["y_pred_test_abs"]

            combined_X = pd.concat([X_train, X_test, future_years], ignore_index=True)            
            combined_y_actual = pd.Series(
                np.concatenate([
                    y_train.values,
                    y_test.values,
                    np.full(len(future_years), np.nan) # NaN para o futuro
                ]),
                index=combined_X.index
            )

            combined_y_pred = pd.Series(
                np.concatenate([y_pred_train, y_pred_test, y_pred_future]),
                index=combined_X.index
            )


            fig_pred = plot_predictions_vs_actuals(
                combined_X,
                combined_y_actual,
                combined_y_pred,
                f"Model Predictions vs. Actuals for {selected_country_name} (with 5-year forecast)",
                st.session_state["target_column"],
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Feature importance
            try:
                fi = st.session_state.get('feature_importance', None)
                if fi is not None and getattr(fi, 'size', 0) > 0:
                    # normalize to pandas Series if necessary
                    if isinstance(fi, pd.DataFrame):
                        fi = pd.Series(fi.iloc[:, 1].values, index=fi.iloc[:, 0].values)
                    
                    st.subheader("🔍 Feature importance (model explainability)")
                    
                    col_toggle, _ = st.columns([0.4, 0.6])
                    hide_lag = col_toggle.checkbox("Hide 'lag_1' (autoregressive feature)", value=True, 
                                        help="Removes the previous year's value impact to show the contribution of other indicators.")
                    
                    fi_to_plot = fi.copy()
                    
                    if hide_lag and 'lag_1' in fi_to_plot.index:
                        fi_to_plot = fi_to_plot.drop('lag_1')
                    
                    if not fi_to_plot.empty:
                        fi_to_plot = (fi_to_plot / fi_to_plot.sum()) * 100
                        
                        st.caption("Relative impact of each indicator on the prediction (%).")
                        fig_fi = plot_feature_importance(fi_to_plot, f"Drivers of {st.session_state.get('target_column', '')}")
                        st.plotly_chart(fig_fi, use_container_width=True)
                    else:
                        st.info("No other features to display besides lag_1.")
                        
                else:
                    st.caption("No feature importance available for this model.")
            except Exception as e:
                st.write("Could not display feature importance:", e)
            # Generate data for all exports upfront
            model = st.session_state["trained_model"]
            X_test = st.session_state["X_test"]

            # Generate future years for prediction (5-year forecast)
            future_years = pd.DataFrame({"year": range(END_YEAR + 1, END_YEAR + 6)})

            # Populate other features using the last known values from X_test
            other_features = X_test.drop(columns=["year"], errors="ignore").columns
            for feature in other_features:
                future_years[feature] = X_test[feature].iloc[-1]

            # Make predictions for future years
            y_pred_future = model.predict(future_years)

            # Create metadata for all exports
            metadata = {
                "model_name": selected_model,
                "target_column": st.session_state["target_column"],
                "country_name": selected_country_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            future_csv_buffer = export_future_predictions(
                future_years, y_pred_future, metadata
            )

            train_test_csv_buffer = export_train_test_data(
                st.session_state["X_train"],
                st.session_state["y_train"],
                st.session_state["X_test"],
                st.session_state["y_test"],
                metadata,
            )

            actual_vs_pred_csv_buffer = export_actual_vs_predicted_data(
                combined_X, combined_y_actual, combined_y_pred, metadata
            )

            # Export model as bytes for download
            model_bytes = save_model(model)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.download_button(
                    label="Export Future Predictions as CSV",
                    data=future_csv_buffer,
                    file_name=f"future_predictions_{selected_country_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_future_predictions",
                )

            with col2:
                st.download_button(
                    label="Export Training/Testing Data as CSV",
                    data=train_test_csv_buffer,
                    file_name=f"train_test_data_{selected_country_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_train_test",
                )

            with col3:
                st.download_button(
                    label="Export Actual vs Predicted Data as CSV",
                    data=actual_vs_pred_csv_buffer,
                    file_name=f"actual_vs_predicted_{selected_country_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="export_actual_vs_predicted",
                )

            with col4:
                st.download_button(
                    label="Export Trained Model",
                    data=model_bytes,
                    file_name=f"model_{selected_model.lower().replace(' ', '_')}_{selected_country_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pkl",
                    mime="application/octet-stream",
                    key="export_trained_model",
                )

with tab3:
    st.header("Model Comparison", divider="rainbow")

    model_history = st.session_state.get("model_history", {})

    if len(model_history) < 2:
        st.warning(
            "⚠️ Train at least two models to compare. Please train models using the 'Model Training & Prediction' tab."
        )
        st.info(
            "💡 Tip: Train different models or the same model with different configurations to compare their performance."
        )
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
                help="This model will be used as the reference for comparison.",
            )

        with col2:
            st.subheader("Model B (Challenger)")
            # Filter out model_a_id from options for model_b
            model_b_options = [mid for mid in model_ids if mid != model_a_id]
            if not model_b_options:
                st.warning("Only one model available. Train another model to compare.")
                model_b_id = None
            else:
                model_b_id = st.selectbox(
                    "Select Model B:",
                    model_b_options,
                    key="model_b_selector",
                    help="This model will be compared against Model A.",
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

            metrics_a = model_a["metrics"]
            metrics_b = model_b["metrics"]

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            ECONOMIC_INDICATORS = {"GDP (current US$)"}
            is_economic = model_a["target_column"] in ECONOMIC_INDICATORS

            with metric_col1:
                st.metric(
                    label="Mean Absolute Error (MAE)",
                    value=(
                        f"${metrics_b['mae']:,.0f}"
                        if is_economic
                        else f"{metrics_b['mae']:,.0f}"
                    ),
                    delta=f"{metrics_b['mae'] - metrics_a['mae']:,.0f}",
                    delta_color="inverse",
                    help="Lower is better. Delta shows Model B - Model A.",
                )

            with metric_col2:
                st.metric(
                    label="R-squared (R²)",
                    value=f"{metrics_b['r2_score']:.3f}",
                    delta=f"{metrics_b['r2_score'] - metrics_a['r2_score']:.3f}",
                    delta_color="normal",
                    help="Higher is better. Delta shows Model B - Model A.",
                )

            with metric_col3:
                st.metric(
                    label="Mean Squared Error (MSE)",
                    value=(
                        f"${metrics_b['mse']:,.0f}"
                        if is_economic
                        else f"{metrics_b['mse']:,.0f}"
                    ),
                    delta=f"{metrics_b['mse'] - metrics_a['mse']:,.0f}",
                    delta_color="inverse",
                    help="Lower is better. Delta shows Model B - Model A.",
                )

            st.write("")

            # Predictions comparison
            st.subheader("🎯 Predictions Comparison")

            pred_col1, pred_col2 = st.columns(2)

            with pred_col1:
                with st.container(border=True):
                    st.markdown(f"**Model A Prediction**")
                    pred_a = model_a["prediction"]
                    formatted_pred_a = (
                        f"${pred_a:,.0f}" if is_economic else f"{pred_a:,.0f}"
                    )
                    st.metric(
                        label=f"Predicted {model_a['target_column']}",
                        value=formatted_pred_a,
                    )

            with pred_col2:
                with st.container(border=True):
                    st.markdown(f"**Model B Prediction**")
                    pred_b = model_b["prediction"]
                    formatted_pred_b = (
                        f"${pred_b:,.0f}" if is_economic else f"{pred_b:,.0f}"
                    )
                    st.metric(
                        label=f"Predicted {model_b['target_column']}",
                        value=formatted_pred_b,
                        delta=(
                            f"{pred_b - pred_a:,.0f}"
                            if is_economic
                            else f"{pred_b - pred_a:,.0f}"
                        ),
                        help="Difference: Model B - Model A",
                    )

            st.write("")

            # Visualizations comparison
            st.subheader("📈 Predictions vs Actuals Comparison")

            viz_col1, viz_col2 = st.columns(2)

            # Helper function to reconstruct plot data (Fixes the scale issue)
            def prepare_comparison_data(model_entry):
                # 1. Retrieve Historical Data
                X_train = model_entry['X_train']
                X_test = model_entry['X_test']
                y_train = model_entry['y_train'] # Already absolute
                y_test = model_entry['y_test']   # Already absolute
                
                # 2. Retrieve Reconstructed Predictions (The Fix)
                # Fallback to direct predict if key missing (backward compatibility)
                if 'y_pred_train_abs' in model_entry:
                    y_pred_train = model_entry['y_pred_train_abs']
                    y_pred_test = model_entry['y_pred_test_abs']
                else:
                    y_pred_train = model_entry['trained_model'].predict(X_train)
                    y_pred_test = model_entry['trained_model'].predict(X_test)
                
                # 3. Calculate Recursive Future
                model = model_entry['trained_model']
                last_features = X_test.iloc[-1]
                X_history = pd.concat([X_train, X_test])
                
                if 'lag_1' in last_features:
                    future_df = make_recursive_future_prediction(
                        model, last_features, X_history, years_ahead=5
                    )
                    y_pred_future = future_df['predicted_value'].values
                    future_years = future_df[['year']]
                else:
                    # Fallback for old models
                    future_years = pd.DataFrame({'year': range(END_YEAR + 1, END_YEAR + 6)})
                    for col in X_test.columns:
                        if col != 'year': future_years[col] = last_features[col]
                    y_pred_future = model.predict(future_years)

                # 4. Combine
                combined_X = pd.concat([X_train, X_test, future_years], ignore_index=True)
                
                combined_y_actual = pd.Series(
                    np.concatenate([y_train.values, y_test.values, np.full(len(future_years), np.nan)]),
                    index=combined_X.index
                )
                
                combined_y_pred = pd.Series(
                    np.concatenate([y_pred_train, y_pred_test, y_pred_future]),
                    index=combined_X.index
                )
                
                return combined_X, combined_y_actual, combined_y_pred

            with viz_col1:
                st.markdown(f"**Model A: {model_a['model_name']}**")
                
                # Use the helper to get corrected absolute data
                cX_a, cYa_a, cYp_a = prepare_comparison_data(model_a)

                fig_a = plot_predictions_vs_actuals(
                    cX_a, cYa_a, cYp_a,
                    f"Model A: {model_a['model_name']}",
                    model_a["target_column"],
                )
                st.plotly_chart(fig_a, use_container_width=True)

                # Feature importance A
                try:
                    fi_a = model_a.get('feature_importance', None)
                    if fi_a is not None and getattr(fi_a, 'size', 0) > 0:
                        if isinstance(fi_a, pd.DataFrame):
                            fi_a = pd.Series(fi_a.iloc[:, 1].values, index=fi_a.iloc[:, 0].values)
                        
                        # Remove Lag for clarity
                        if 'lag_1' in fi_a.index:
                            fi_a = fi_a.drop('lag_1')
                        
                        # Normalize to %
                        if not fi_a.empty:
                            fi_a = (fi_a / fi_a.sum()) * 100
                            st.markdown("**Feature importance (%) - Model A**")
                            fig_fi_a = plot_feature_importance(fi_a, "")
                            fig_fi_a.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250)
                            st.plotly_chart(fig_fi_a, use_container_width=True)
                except Exception as e:
                    st.caption(f"Error showing feature importance: {e}")

            with viz_col2:
                st.markdown(f"**Model B: {model_b['model_name']}**")
                
                # Use the helper to get corrected absolute data
                cX_b, cYa_b, cYp_b = prepare_comparison_data(model_b)

                fig_b = plot_predictions_vs_actuals(
                    cX_b, cYa_b, cYp_b,
                    f"Model B: {model_b['model_name']}",
                    model_b["target_column"],
                )
                st.plotly_chart(fig_b, use_container_width=True)

                # Feature importance B
                try:
                    fi_b = model_b.get('feature_importance', None)
                    if fi_b is not None and getattr(fi_b, 'size', 0) > 0:
                        if isinstance(fi_b, pd.DataFrame):
                            fi_b = pd.Series(fi_b.iloc[:, 1].values, index=fi_b.iloc[:, 0].values)
                        
                        # Remove Lag for clarity
                        if 'lag_1' in fi_b.index:
                            fi_b = fi_b.drop('lag_1')
                            
                        # Normalize to %
                        if not fi_b.empty:
                            fi_b = (fi_b / fi_b.sum()) * 100
                            st.markdown("**Feature importance (%) - Model B**")
                            fig_fi_b = plot_feature_importance(fi_b, "")
                            fig_fi_b.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=250)
                            st.plotly_chart(fig_fi_b, use_container_width=True)
                except Exception as e:
                    st.caption(f"Error showing feature importance: {e}")              
