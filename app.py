import streamlit as st
import pandas as pd
import numpy as np
from src.data_processing import fetch_world_bank_data
from src.model_training import select_model, prepare_data, train_model, evaluate_model, make_prediction, MODELS
from src.visualization import plot_indicator_trend, plot_predictions_vs_actuals

st.set_page_config(
    page_title="Economic Indicator Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            
            if 'country' in country_data.columns:
                country_data_for_training = country_data.drop(columns=['country'])
            else:
                country_data_for_training = country_data

            X_train, X_test, y_train, y_test = prepare_data(
                country_data_for_training, selected_target_column)
            st.session_state['X_test'], st.session_state['y_test'] = X_test, y_test

            internal_model_name = MODEL_NAME_MAP[selected_model]
            unfitted_model = select_model(internal_model_name)
            model = train_model(unfitted_model, X_train, y_train)
            st.session_state['trained_model'] = model

            metrics = evaluate_model(model, X_test, y_test)
            st.session_state['metrics'] = metrics

            # Prepare features for the single next-year prediction (END_YEAR + 1)
            next_year_features = pd.DataFrame({'year': [END_YEAR + 1]})
            
            # Determine the source for other features: X_test_df if not empty, else X_train
            features_source_df = X_test if not X_test.empty else X_train

            # Populate other features using the last known values from the chosen source
            other_features = features_source_df.drop(columns=['year'], errors='ignore').columns
            for feature in other_features:
                next_year_features[feature] = features_source_df[feature].iloc[-1]

            prediction = make_prediction(model, next_year_features)
            st.session_state['prediction'] = prediction[0]
            st.session_state['selected_target_column'] = selected_target_column
            
            st.success("Model trained successfully!")

st.title("📈 Economic Indicator Prediction Dashboard")
st.markdown(f"Currently analyzing: **{selected_country_name}**")

tab1, tab2 = st.tabs(["📊 Data Exploration & Visualization",
                     "🧠 Model Training & Prediction"])

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
        is_economic = st.session_state['selected_target_column'] in ECONOMIC_INDICATORS

        with st.container(border=True):
            st.subheader(f"🔮 Prediction for {st.session_state['selected_target_column']} in {END_YEAR + 1}")
            pred_value = st.session_state['prediction']
            
            if is_economic:
                formatted_pred = f"${pred_value:,.0f}"
            else:
                formatted_pred = f"{pred_value:,.0f}"

            st.metric(
                label=f"Predicted {st.session_state['selected_target_column']}",
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
            model = st.session_state['trained_model']
            X_test_df = st.session_state['X_test']
            y_test_series = st.session_state['y_test']
            y_pred_test = model.predict(X_test_df)

            # Get the last year from the training data (which is END_YEAR)
            last_training_year = END_YEAR

            # Generate future years for prediction
            future_years = pd.DataFrame({'year': range(last_training_year + 1, last_training_year + 6)})

            # For other features, use the last known values from X_test_df
            # This assumes other features are not time-dependent or their last values are good enough for future prediction
            other_features = X_test_df.drop(columns=['year'], errors='ignore').columns
            for feature in other_features:
                # Use the last known value for each other feature
                future_years[feature] = X_test_df[feature].iloc[-1]

            # Make predictions for future years
            y_pred_future = model.predict(future_years)

            # Combine X_train, X_test, and future_years for plotting
            # Ensure X_train and X_test are sorted by year before concatenation
            # (prepare_data already sorts the original df, so X_train and X_test should be sorted)
            combined_X = pd.concat([X_train, X_test_df, future_years], ignore_index=True)

            # Combine y_train (actuals for training data), y_test (actuals for test data), and NaNs for future years
            combined_y_actual = pd.Series(
                np.concatenate([y_train.values, y_test_series.values, np.full(len(future_years), np.nan)]),
                index=combined_X.index
            )

            # y_pred_test is already calculated based on X_test_df
            # For y_pred_train, we need to predict on X_train
            y_pred_train = model.predict(X_train)

            # Combine y_pred_train, y_pred_test, and y_pred_future for plotting
            combined_y_pred = pd.Series(
                np.concatenate([y_pred_train, y_pred_test, y_pred_future]),
                index=combined_X.index
            )

            fig_pred = plot_predictions_vs_actuals(
                combined_X,
                combined_y_actual,
                combined_y_pred,
                f"Model Predictions vs. Actuals for {selected_country_name} (with 5-year forecast)",
                st.session_state['selected_target_column']
            )
            st.plotly_chart(fig_pred, use_container_width=True)

