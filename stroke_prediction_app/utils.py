import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder

def load_model(model_path):
    """
    Load a model from the specified path.
    
    Args:
        model_path (str): Path to the saved model file
    
    Returns:
        object: The loaded model
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_input(input_data):
    """
    Preprocess user input to match the expected format for the model.
    
    Args:
        input_data (dict): Dictionary containing user input values
    
    Returns:
        pandas.DataFrame: Preprocessed data ready for prediction
    """
    # Convert input dictionary to pandas DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables to match training data encoding
    # Gender encoding
    gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
    input_df['gender'] = input_df['gender'].map(gender_map)
    
    # Ever married encoding
    ever_married_map = {'Yes': 1, 'No': 0}
    input_df['ever_married'] = input_df['ever_married'].map(ever_married_map)
    
    # Work type encoding
    work_type_map = {'Private': 0, 'Self-employed': 1, 'Government job': 2, 
                     'Children': 3, 'Never worked': 4}
    input_df['work_type'] = input_df['work_type'].map(work_type_map)
    
    # Residence type encoding
    residence_type_map = {'Urban': 1, 'Rural': 0}
    input_df['Residence_type'] = input_df['Residence_type'].map(residence_type_map)
    
    # Smoking status encoding
    smoking_status_map = {'formerly smoked': 1, 'never smoked': 2, 
                          'smokes': 3, 'Unknown': 0}
    input_df['smoking_status'] = input_df['smoking_status'].map(smoking_status_map)
    
    # FIXED: Use the correct column order from the model logs
    expected_column_order = [
        'gender', 'age', 'hypertension', 'heart_disease',
        'ever_married', 'work_type', 'Residence_type',
        'avg_glucose_level', 'bmi', 'smoking_status'
    ]
    
    # Reorder columns to match the model's expected order
    input_df = input_df[expected_column_order]
    
    return input_df

def calculate_performance_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate performance metrics for model evaluation.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_prob (array-like, optional): Predicted probabilities for positive class
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, cohen_kappa_score,
        matthews_corrcoef
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }
    
    if y_prob is not None:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def plot_model_performance(metrics, model_name):
    """
    Create visualizations for model performance.
    
    Args:
        metrics (dict): Dictionary containing performance metrics
        model_name (str): Name of the model
    
    Returns:
        tuple: Tuple containing Matplotlib figures for different visualizations
    """
    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt="d", cmap="Blues",
               xticklabels=["No Stroke", "Stroke"],
               yticklabels=["No Stroke", "Stroke"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix - {model_name}")
    
    # Create bar chart for key metrics
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'kappa']
    metric_values = [metrics.get(m, 0) for m in key_metrics]
    
    fig_metrics, ax_metrics = plt.subplots(figsize=(10, 6))
    ax_metrics.bar(key_metrics, metric_values)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.set_title(f"Performance Metrics - {model_name}")
    ax_metrics.set_ylabel("Score")
    
    for i, v in enumerate(metric_values):
        if v is not None:
            ax_metrics.text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    return fig_cm, fig_metrics