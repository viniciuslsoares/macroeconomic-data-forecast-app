import sys
import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

# --- Imports: Model Layer ---
from src.model.model_registry import get_model_names
from src.model.model_training import (
    run_training_pipeline, 
    save_model, 
    evaluate_loaded_model, 
    make_recursive_future_prediction
)
from src.model.data_processing import fetch_world_bank_data

# --- Imports: View Layer ---
from src.view.visualization import (
    plot_indicator_trend, 
    plot_predictions_vs_actuals, 
    plot_feature_importance
)
from src.view.ui_components import (
    render_sidebar, 
    render_metrics_section, 
    render_prediction_banner, 
    apply_custom_styles
)
from src.view.export_utils import (
    export_train_test_data, 
    export_actual_vs_predicted_data, 
    export_future_predictions
)

# --- Application Configuration ---
st.set_page_config(
    page_title="Economic Indicator Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply CSS styles from View
apply_custom_styles()

# Initialize Session State
if "model_history" not in st.session_state:
    st.session_state["model_history"] = {}

# --- Constants ---
COUNTRIES = {"Brazil": "BRA", "Canada": "CAN"}
INDICATORS = {
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "SP.POP.TOTL": "Population, total",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
}
START_YEAR = 2000
END_YEAR = 2018

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def get_data():
    """Fetches data from Model layer (cached)."""
    with st.spinner("⏳ Fetching and preprocessing data from World Bank API..."):
        return fetch_world_bank_data(
            list(COUNTRIES.values()), START_YEAR, END_YEAR, INDICATORS
        )

def save_training_results_to_state(results, country_name, model_name, target_column):
    """Helper to persist results and history."""
    # 1. Update current session state
    for key, value in results.items():
        st.session_state[key] = value
    
    st.session_state["selected_country_name"] = country_name
    
    # 2. Add to history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_id = f"{model_name} - {target_column} - {timestamp}"
    
    history_entry = results.copy()
    history_entry.update({
        "model_name": model_name,
        "target_column": target_column,
        "country_name": country_name,
        "timestamp": timestamp
    })
    
    st.session_state["model_history"][model_id] = history_entry

# --- Main Controller Logic ---

def main():
    # 1. Data Loading (Model)
    main_data = get_data()
    
    # 2. Render Sidebar & Get User Input (View)
    user_input = render_sidebar(COUNTRIES, get_model_names(), INDICATORS)
    
    # Identify selected country index
    country_names = list(COUNTRIES.keys())
    selected_country_index = country_names.index(user_input["country_name"])

    # 3. Handle Event: Train Model
    if user_input["train_clicked"]:
        with st.spinner("Training model... Please wait."):
            country_data = main_data[selected_country_index].copy()
            
            # Call Model Pipeline
            results = run_training_pipeline(
                country_data=country_data,
                target_column=user_input["target"],
                model_name=user_input["model"],
                end_year=END_YEAR,
            )
            
            save_training_results_to_state(
                results, user_input["country_name"], user_input["model"], user_input["target"]
            )
            st.success("Model trained successfully!")

    # 4. Handle Event: Upload Model
    if user_input["uploaded_file"]:
        try:
            model = joblib.load(io.BytesIO(user_input["uploaded_file"].getvalue()))
            country_data = main_data[selected_country_index].copy()
            
            # Call Model Evaluation Logic
            results = evaluate_loaded_model(
                model=model,
                country_data=country_data,
                target_column=user_input["target"],
                end_year=END_YEAR,
            )
            
            save_training_results_to_state(
                results, user_input["country_name"], "Loaded Model", user_input["target"]
            )
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    # 5. Render Main Dashboard (View)
    st.title("📈 Economic Indicator Prediction Dashboard")
    st.markdown(f"Currently analyzing: **{user_input['country_name']}**")

    tab1, tab2, tab3 = st.tabs([
        "📊 Data Exploration", 
        "🧠 Model Prediction", 
        "⚖️ Model Comparison"
    ])

    # --- TAB 1: Data Exploration ---
    with tab1:
        st.header("Initial Data Analysis", divider="rainbow")
        
        with st.container(border=True):
            st.subheader("Preprocessed Data Table")
            data_table = main_data[selected_country_index].tail(10)
            st.dataframe(data_table)
            
            # Export Data Button
            csv_buffer = io.BytesIO()
            data_table.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Export Data as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"data_{user_input['country_name']}.csv",
                mime="text/csv"
            )

        st.write("")
        
        # Trend Visualization
        with st.container(border=True):
            st.subheader("Indicator Trends")
            indicator_to_plot = st.selectbox(
                "Select indicator:", list(INDICATORS.values()), key="viz_indicator"
            )
            if indicator_to_plot:
                fig = plot_indicator_trend(
                    main_data[selected_country_index], 
                    indicator_to_plot, 
                    f"{indicator_to_plot} Trend"
                )
                st.plotly_chart(fig, width='stretch')

    # --- TAB 2: Model Prediction ---
    with tab2:
        st.header("Machine Learning Results", divider="rainbow")

        if "metrics" in st.session_state:
            is_eco = "GDP" in st.session_state["target_column"]
            
            # 1. Prediction Banner (View Component)
            render_prediction_banner(
                st.session_state["prediction"], 
                st.session_state["target_column"], 
                END_YEAR + 1, 
                is_eco
            )

            st.write("")

            # 2. Metrics Section (View Component)
            model_details = {
                "model": st.session_state.get("selected_model_name", user_input["model"]),
                "country": st.session_state.get("selected_country_name", user_input["country_name"]),
                "target": st.session_state["target_column"]
            }
            render_metrics_section(st.session_state["metrics"], model_details, is_eco)

            st.write("")
            # 3. Main Chart: Actual vs Predicted
            with st.container(border=True):
                st.subheader("🎯 Actual vs. Predicted Values")
                
                # Retrieve data from state
                model = st.session_state["trained_model"]
                X_train = st.session_state["X_train"]
                y_train = st.session_state["y_train"] # Absolute values
                X_test = st.session_state["X_test"]
                y_test = st.session_state["y_test"]   # Absolute values
                
                # Prepare Recursive Future Prediction
                last_features = X_test.iloc[-1]
                X_history = pd.concat([X_train, X_test])
                
                if 'lag_1' in last_features:
                    future_df = make_recursive_future_prediction(
                        model, last_features, X_history, years_ahead=5
                    )
                    y_pred_future = future_df['predicted_value'].values
                    future_years = future_df[['year']]
                else:
                    st.warning("⚠️ Legacy model detected. Using static prediction.")
                    future_years = pd.DataFrame({'year': range(END_YEAR + 1, END_YEAR + 6)})
                    for c in X_test.columns:
                        if c != 'year': future_years[c] = last_features[c]
                    y_pred_future = model.predict(future_years)

                # Use reconstructed absolute predictions from pipeline
                y_pred_train = st.session_state.get("y_pred_train_abs", model.predict(X_train))
                y_pred_test = st.session_state.get("y_pred_test_abs", model.predict(X_test))

                # Combine for plotting
                combined_X = pd.concat([X_train, X_test, future_years], ignore_index=True)
                combined_y_actual = pd.Series(
                    np.concatenate([y_train.values, y_test.values, np.full(len(future_years), np.nan)]),
                    index=combined_X.index
                )
                combined_y_pred = pd.Series(
                    np.concatenate([y_pred_train, y_pred_test, y_pred_future]),
                    index=combined_X.index
                )

                fig_pred = plot_predictions_vs_actuals(
                    combined_X, combined_y_actual, combined_y_pred,
                    f"Forecast for {user_input['country_name']}",
                    st.session_state["target_column"]
                )
                st.plotly_chart(fig_pred, width='stretch')

            # 4. Feature Importance
            fi = st.session_state.get('feature_importance', None)
            if fi is not None and getattr(fi, 'size', 0) > 0:
                if isinstance(fi, pd.DataFrame):
                    fi = pd.Series(fi.iloc[:, 1].values, index=fi.iloc[:, 0].values)
                
                st.subheader("🔍 Feature Importance")
                col_toggle, _ = st.columns([0.4, 0.6])
                hide_lag = col_toggle.checkbox("Hide 'lag_1'", value=True)
                
                if hide_lag and 'lag_1' in fi.index:
                    fi = fi.drop('lag_1')
                
                if not fi.empty:
                    # Normalize to %
                    fi = (fi / fi.sum()) * 100
                    fig_fi = plot_feature_importance(fi, "Key Drivers (%)")
                    st.plotly_chart(fig_fi, width='stretch')
                else:
                    st.info("No features to display.")

            # 5. Exports
            st.divider()
            st.subheader("📥 Exports")
            col1, col2, col3, col4 = st.columns(4)
            
            metadata = {
                "model": user_input["model"], 
                "country": user_input["country_name"],
                "timestamp": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Helper to generate CSVs
            with col1:
                st.download_button(
                    "Future Predictions (CSV)",
                    data=export_future_predictions(future_years, y_pred_future, metadata),
                    file_name="future_preds.csv", mime="text/csv"
                )
            with col2:
                st.download_button(
                    "Train/Test Data (CSV)",
                    data=export_train_test_data(X_train, y_train, X_test, y_test, metadata),
                    file_name="train_test_data.csv", mime="text/csv"
                )
            with col3:
                st.download_button(
                    "Actual vs Pred (CSV)",
                    data=export_actual_vs_predicted_data(combined_X, combined_y_actual, combined_y_pred, metadata),
                    file_name="actual_vs_pred.csv", mime="text/csv"
                )
            with col4:
                st.download_button(
                    "Trained Model (PKL)",
                    data=save_model(model),
                    file_name="model.pkl", mime="application/octet-stream"
                )

        else:
            st.info("👈 Please train a model in the sidebar to see results.")

    # --- TAB 3: Comparison ---
    with tab3:
        st.header("Model Comparison", divider="rainbow")
        
        history = st.session_state["model_history"]
        
        if len(history) < 2:
            st.warning("⚠️ Train at least two models to compare.")
        else:
            model_ids = list(history.keys())
            c1, c2 = st.columns(2)
            
            with c1:
                id_a = st.selectbox("Model A (Reference)", model_ids, key="sel_a")
            with c2:
                # Filter out Model A from options
                opts_b = [m for m in model_ids if m != id_a]
                id_b = st.selectbox("Model B (Challenger)", opts_b, key="sel_b") if opts_b else None
                
            if id_b:
                mod_a = history[id_a]
                mod_b = history[id_b]
                
                # Metrics Comparison
                st.subheader("📊 Metrics Delta")
                mc1, mc2, mc3 = st.columns(3)
                
                delta_mae = mod_b["metrics"]["mae"] - mod_a["metrics"]["mae"]
                delta_r2 = mod_b["metrics"]["r2_score"] - mod_a["metrics"]["r2_score"]
                delta_mse = mod_b["metrics"]["mse"] - mod_a["metrics"]["mse"]
                
                mc1.metric("MAE", f"{mod_b['metrics']['mae']:.2f}", f"{delta_mae:.2f}", delta_color="inverse")
                mc2.metric("R²", f"{mod_b['metrics']['r2_score']:.3f}", f"{delta_r2:.3f}")
                mc3.metric("MSE", f"{mod_b['metrics']['mse']:.2f}", f"{delta_mse:.2f}", delta_color="inverse")
                
                # Visual Comparison
                st.divider()
                st.subheader("📈 Visual Comparison")
                vc1, vc2 = st.columns(2)
                
                def get_plot_data(entry):
                    # Helper to reconstruct plotting data from history entry
                    model = entry["trained_model"]
                    X_tr, X_te = entry["X_train"], entry["X_test"]
                    y_tr, y_te = entry["y_train"], entry["y_test"]
                    
                    # Use stored absolutes or recalculate
                    yp_tr = entry.get("y_pred_train_abs", model.predict(X_tr))
                    yp_te = entry.get("y_pred_test_abs", model.predict(X_te))
                    
                    # Future
                    last_f = X_te.iloc[-1]
                    hist = pd.concat([X_tr, X_te])
                    if 'lag_1' in last_f:
                        f_df = make_recursive_future_prediction(model, last_f, hist, 5)
                        yp_fut = f_df['predicted_value'].values
                        fut_yrs = f_df[['year']]
                    else:
                        fut_yrs = pd.DataFrame({'year': range(END_YEAR+1, END_YEAR+6)})
                        for c in X_te.columns: 
                            if c!='year': fut_yrs[c] = last_f[c]
                        yp_fut = model.predict(fut_yrs)
                        
                    cX = pd.concat([X_tr, X_te, fut_yrs], ignore_index=True)
                    cY = pd.Series(np.concatenate([y_tr, y_te, np.full(len(fut_yrs), np.nan)]), index=cX.index)
                    cpY = pd.Series(np.concatenate([yp_tr, yp_te, yp_fut]), index=cX.index)
                    return cX, cY, cpY

                # --- Helper to Render Feature Importance in Comparision ---
                def render_comparison_fi(model_entry, key_suffix):
                    fi = model_entry.get('feature_importance', None)
                    if fi is not None and getattr(fi, 'size', 0) > 0:
                        st.markdown("##### 🔍 Key Drivers")
                        
                        if isinstance(fi, pd.DataFrame):
                            fi = pd.Series(fi.iloc[:, 1].values, index=fi.iloc[:, 0].values)
                        
                        hide_lag = st.checkbox("Hide 'lag_1'", value=True, key=f"hide_lag_{key_suffix}")
                        
                        fi_plot = fi.copy()
                        if hide_lag and 'lag_1' in fi_plot.index:
                            fi_plot = fi_plot.drop('lag_1')
                        
                        if not fi_plot.empty:
                            fi_plot = (fi_plot / fi_plot.sum()) * 100
                            fig = plot_feature_importance(fi_plot, "")
                            fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0))
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.caption("No features to display (lag_1 hidden).")
                    else:
                        st.caption("No feature importance available.")

                # Model A
                with vc1:
                    st.caption(f"**Model A: {mod_a['model_name']}**")
                    cX, cY, cpY = get_plot_data(mod_a)
                    fig_a = plot_predictions_vs_actuals(cX, cY, cpY, "", mod_a["target_column"])
                    st.plotly_chart(fig_a, width='stretch')
                    
                    st.divider()
                    render_comparison_fi(mod_a, "A")
                    
                # Model B
                with vc2:
                    st.caption(f"**Model B: {mod_b['model_name']}**")
                    cX, cY, cpY = get_plot_data(mod_b)
                    fig_b = plot_predictions_vs_actuals(cX, cY, cpY, "", mod_b["target_column"])
                    st.plotly_chart(fig_b, width='stretch')
                    
                    st.divider()
                    render_comparison_fi(mod_b, "B")

if __name__ == "__main__":
    main()