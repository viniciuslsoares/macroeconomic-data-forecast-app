import pandas as pd
import io
from typing import Dict, Any
import streamlit as st


def export_train_test_data(X_train: pd.DataFrame, y_train: pd.Series, 
                          X_test: pd.DataFrame, y_test: pd.Series, 
                          metadata: Dict[str, Any]) -> io.BytesIO:
    """
    Export training and testing data as CSV.
    
    Args:
        X_train: Training features DataFrame
        y_train: Training target Series
        X_test: Test features DataFrame
        y_test: Test target Series
        metadata: Dictionary containing metadata about the model and data
        
    Returns:
        BytesIO object containing the CSV data
    """
    # Combine training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df['data_type'] = 'train'
    
    # Combine test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df['data_type'] = 'test'
    
    # Combine all data
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Add metadata as comments at the top
    metadata_lines = [
        f"# Model: {metadata.get('model_name', 'Unknown')}",
        f"# Target Column: {metadata.get('target_column', 'Unknown')}",
        f"# Country: {metadata.get('country_name', 'Unknown')}",
        f"# Exported on: {metadata.get('timestamp', 'Unknown')}",
        ""
    ]
    
    csv_buffer = io.BytesIO()
    metadata_str = "\n".join(metadata_lines)
    csv_buffer.write(metadata_str.encode('utf-8'))
    
    # Write the actual CSV data
    combined_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def export_actual_vs_predicted_data(combined_X: pd.DataFrame, 
                                   combined_y_actual: pd.Series, 
                                   combined_y_pred: pd.Series, 
                                   metadata: Dict[str, Any]) -> io.BytesIO:
    """
    Export actual vs predicted data as CSV.
    
    Args:
        combined_X: Combined features DataFrame (train + test + future)
        combined_y_actual: Combined actual values Series
        combined_y_pred: Combined predicted values Series
        metadata: Dictionary containing metadata about the model and data
        
    Returns:
        BytesIO object containing the CSV data
    """
    # Create results DataFrame
    results_df = pd.DataFrame({
        'year': combined_X['year'],
        'actual_value': combined_y_actual,
        'predicted_value': combined_y_pred,
        'data_type': ['actual' if not pd.isna(actual) else 'future' 
                     for actual in combined_y_actual]
    })
    
    # Add metadata as comments at the top
    metadata_lines = [
        f"# Model: {metadata.get('model_name', 'Unknown')}",
        f"# Target Column: {metadata.get('target_column', 'Unknown')}",
        f"# Country: {metadata.get('country_name', 'Unknown')}",
        f"# Exported on: {metadata.get('timestamp', 'Unknown')}",
        f"# Note: 'actual' data includes training and test sets, 'future' data includes predictions",
        ""
    ]
    
    csv_buffer = io.BytesIO()
    metadata_str = "\n".join(metadata_lines)
    csv_buffer.write(metadata_str.encode('utf-8'))
    
    # Write the actual CSV data
    results_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def export_future_predictions(future_years: pd.DataFrame, 
                             y_pred_future: pd.Series, 
                             metadata: Dict[str, Any]) -> io.BytesIO:
    """
    Export future predictions as CSV.
    
    Args:
        future_years: DataFrame containing future years
        y_pred_future: Series containing future predictions
        metadata: Dictionary containing metadata about the model and data
        
    Returns:
        BytesIO object containing the CSV data
    """
    # Create future predictions DataFrame
    future_df = pd.DataFrame({
        'year': future_years['year'],
        'predicted_value': y_pred_future
    })
    
    # Add metadata as comments at the top
    metadata_lines = [
        f"# Model: {metadata.get('model_name', 'Unknown')}",
        f"# Target Column: {metadata.get('target_column', 'Unknown')}",
        f"# Country: {metadata.get('country_name', 'Unknown')}",
        f"# Exported on: {metadata.get('timestamp', 'Unknown')}",
        f"# Note: Future predictions for 5 years ahead",
        ""
    ]
    
    csv_buffer = io.BytesIO()
    metadata_str = "\n".join(metadata_lines)
    csv_buffer.write(metadata_str.encode('utf-8'))
    
    # Write the actual CSV data
    future_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return csv_buffer


def create_csv_download_button(data: io.BytesIO, filename: str, button_label: str, key: str) -> None:
    """
    Create a Streamlit download button for CSV data.
    
    Args:
        data: BytesIO object containing CSV data
        filename: Name of the file to download
        button_label: Label for the download button
        key: Unique key for the button
    """
    st.download_button(
        label=button_label,
        data=data,
        file_name=filename,
        mime="text/csv",
        key=key
    )
