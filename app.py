# File: app.py
import streamlit as st
import pandas as pd
from src.data_processing import fetch_world_bank_data
from src.model_training import select_model, prepare_data, train_model, evaluate_model, make_prediction, MODELS
from src.visualization import plot_indicator_trend, plot_predictions_vs_actuals

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Economic Indicator Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Constants & Configuration ---
# (No changes here, logic remains the same)
COUNTRIES = {"Brazil": "BRA", "Canada": "CAN"}
INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP (current US$)',
    'SP.POP.TOTL': 'Population, total',
    'SP.DYN.LE00.IN': 'Life expectancy at birth, total (years)',
    'IT.NET.USER.ZS': 'Individuals using the Internet (% of population)'
}
START_YEAR = 2000
END_YEAR = 2023


# --- 3. Data Loading ---
@st.cache_data
def load_data():
    """Fetches and preprocesses data from the World Bank. Caches the result."""
    with st.spinner("Fetching and preprocessing data from the World Bank API... This may take a moment."):
        data = fetch_world_bank_data(
            list(COUNTRIES.values()), START_YEAR, END_YEAR, INDICATORS)
    return data


# --- Load data once and cache it ---
main_data = load_data()


# --- 4. Sidebar for User Inputs ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.title("⚙️ Configuration Panel")
    st.markdown(
        "Select the country and the machine learning model to be used for the prediction.")

    country_names = list(COUNTRIES.keys())
    selected_country_name = st.selectbox(
        "Select Country", country_names)
    selected_country_index = country_names.index(selected_country_name)

    selected_model = st.selectbox("Select Model", list(MODELS.keys()))

    selected_target_column = st.selectbox(
        "Select Target Column", list(INDICATORS.values()))

    st.markdown("---")

    # The button click triggers the training and saves results to session_state
    if st.button("Train Model & Predict", type="primary", use_container_width=True):
        with st.spinner("Training model... Please wait."):

            # Map user-friendly names from the UI to the internal names expected by `select_model`
            MODEL_NAME_MAP = {
                "Linear Regression": "LinearRegression",
                "Random Forest": "RandomForestRegressor",
                "Gradient Boosting": "GradientBoostingRegressor",
            }
            
            country_data = main_data[selected_country_index].copy()
            
            # The new `prepare_data` expects a DataFrame with only features and target.
            if 'country' in country_data.columns:
                country_data_for_training = country_data.drop(columns=['country'])
            else:
                country_data_for_training = country_data

            # Call the new `prepare_data` function which has a different signature.
            X_train, X_test, y_train, y_test = prepare_data(
                country_data_for_training, selected_target_column)
            st.session_state['X_test'], st.session_state['y_test'] = X_test, y_test

            internal_model_name = MODEL_NAME_MAP[selected_model]
            unfitted_model = select_model(internal_model_name)
            model = train_model(unfitted_model, X_train, y_train)
            st.session_state['trained_model'] = model

            metrics = evaluate_model(model, X_test, y_test)
            st.session_state['metrics'] = metrics

            features_for_prediction = X_train.columns
            last_known_features = country_data.sort_values(
                by='year').iloc[[-1]][features_for_prediction]
            
            prediction = make_prediction(model, last_known_features)
            st.session_state['prediction'] = prediction[0]
            st.session_state['selected_target_column'] = selected_target_column
            

            st.success("Model trained successfully!")


# --- 5. Main Application Body ---
st.title("📈 Economic Indicator Prediction Dashboard")
st.markdown(f"Currently analyzing: **{selected_country_name}**")

# UI IMPROVEMENT: Using tabs to organize the content into logical sections.
tab1, tab2 = st.tabs(["📊 Data Exploration & Visualization",
                     "🧠 Model Training & Prediction"])

# --- Tab 1: Data Exploration ---
with tab1:
    st.header("Initial Data Analysis", divider='rainbow')

    # UI IMPROVEMENT: Using a container to create a visual "box" for the data table.
    with st.container(border=True):
        st.subheader("Preprocessed Data Table")
        st.caption(
            f"Displaying the last 10 years of available data for {selected_country_name}. The model will be trained on this dataset.")
        st.dataframe(main_data[selected_country_index].tail(10))

    st.write("")  # Adds vertical space

    # UI IMPROVEMENT: Another container for the interactive plot.
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

# --- Tab 2: Model Training & Prediction ---
with tab2:
    st.header("Machine Learning Results", divider='rainbow')

    # UI IMPROVEMENT: Check if the model has been trained before showing results.
    if 'metrics' not in st.session_state:
        st.info(
            "Please click the 'Train Model & Predict' button in the sidebar to see the results.")
        st.image(
            "https://media1.tenor.com/m/y2uA8hd_3tEAAAAC/what-are-you-waiting-for-do-it.gif", width=300)
    else:
        # UI IMPROVEMENT: A dedicated container for the main prediction result.
        with st.container(border=True):
            st.subheader(f"🔮 Prediction for {st.session_state['selected_target_column']} in {END_YEAR + 1}")
            pred_value = st.session_state['prediction']
            st.metric(
                label=f"Predicted {st.session_state['selected_target_column']}",
                value=f"${pred_value:,.0f}",
                help="This is the model's prediction for the next year based on the latest available data."
            )

        st.write("")  # Adds vertical space

        # UI IMPROVEMENT: Two columns for a balanced layout.
        col1, col2 = st.columns(2)

        with col1:
            # UI IMPROVEMENT: Container for model performance metrics.
            with st.container(border=True):
                st.subheader("📊 Model Performance Metrics")
                st.caption(
                    "These metrics evaluate the model's accuracy on the unseen test dataset.")
                metrics = st.session_state['metrics']
                # UPDATED: Keys are now lowercase as returned by the new function
                st.metric("Mean Absolute Error (MAE)",
                          f"${metrics['mae']:,.0f}")
                st.metric("R-squared (R²)", f"{metrics['r2_score']:.3f}")

        with col2:
            # UI IMPROVEMENT: Container for showing the chosen model.
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

        st.write("")  # Adds vertical space

        # UI IMPROVEMENT: Final container for the comparison plot.
        with st.container(border=True):
            st.subheader("🎯 Actual vs. Predicted Values (on Test Set)")
            st.caption("This chart helps to visually assess the model's performance by comparing its predictions against the actual historical data it did not see during training.")
            model = st.session_state['trained_model']
            y_pred = model.predict(st.session_state['X_test'])
            fig_pred = plot_predictions_vs_actuals(st.session_state['y_test'], pd.Series(
                y_pred, index=st.session_state['y_test'].index), f"Model Predictions vs. Actuals for {selected_country_name}", st.session_state['selected_target_column'])
            st.plotly_chart(fig_pred, use_container_width=True)

