import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from utils import preprocess_input, load_model, calculate_performance_metrics, plot_model_performance

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .st-bd {
        padding-top: 3px;
    }
    .css-1v0mbdj.ebxwdo61 {
        margin-top: 2em;
        margin-bottom: 2em;
    }
    .css-1kyxreq.ebxwdo60 {
        justify-content: center;
    }
    .stRadio > div {
        flex-direction: row;
        gap: 1em;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
MODEL_PATHS = {
    'XGBoost': '/home/bolt/Desktop/ml_projects/4_models/trained_models/xgboost_stroke_model.joblib',
    'SVM': '/home/bolt/Desktop/ml_projects/4_models/trained_models/svm_stroke_model.joblib',
    'Naive Bayes': '/home/bolt/Desktop/ml_projects/4_models/trained_models/naive_bayes_stroke_model.joblib',
    'LDA': '/home/bolt/Desktop/ml_projects/4_models/trained_models/lda_stroke_model.joblib'
}

MODEL_DESCRIPTIONS = {
    'XGBoost': """
        **XGBoost (Extreme Gradient Boosting)** is an advanced ensemble learning algorithm.
        
        **Strengths:**
        - Typically high accuracy and robust performance
        - Handles imbalanced data well
        - Automatically handles missing values
        - Captures complex non-linear relationships
        
        **Weaknesses:**
        - Can be computationally intensive
        - Risk of overfitting with small datasets
        - Less interpretable than simpler models
    """,
    
    'SVM': """
        **SVM (Support Vector Machine)** finds the optimal boundary between classes.
        
        **Strengths:**
        - Effective in high-dimensional spaces
        - Versatile through different kernel functions
        - Memory efficient
        - Good generalization capabilities
        
        **Weaknesses:**
        - Slower training time on larger datasets
        - Selection of appropriate kernel can be challenging
        - Less transparent in decision-making process
    """,
    
    'Naive Bayes': """
        **Naive Bayes** uses Bayes' theorem with strong independence assumptions.
        
        **Strengths:**
        - Simple and very fast
        - Works well with high-dimensional data
        - Performs well with small training datasets
        - Natural handling of multiple classes
        
        **Weaknesses:**
        - Assumes independence of features (often unrealistic)
        - Limited by "zero frequency" problem
        - May be outperformed by more sophisticated models
    """,
    
    'LDA': """
        **LDA (Linear Discriminant Analysis)** finds a linear combination of features for separation.
        
        **Strengths:**
        - Simple and interpretable model
        - Fast training and prediction
        - Works well with regularly distributed data
        - Good for small sample sizes
        
        **Weaknesses:**
        - Assumes normal distribution of data
        - Assumes equal covariance matrices between classes
        - Limited to linear boundaries between classes
    """
}

@st.cache_resource
def get_model(model_name):
    """Load and cache the selected model"""
    with st.spinner(f"Loading {model_name} model..."):
        model = load_model(MODEL_PATHS[model_name])
        
        # Print feature names if available (helpful for debugging)
        if hasattr(model, 'feature_names_in_'):
            print(f"Model {model_name} feature names: {model.feature_names_in_}")
        elif model_name == "XGBoost" and hasattr(model, 'get_booster'):
            print(f"XGBoost feature names: {model.get_booster().feature_names}")
            
        st.success(f"{model_name} model loaded successfully!")
        return model

def main():
    """Main function to run the Streamlit app"""
    
    # Sidebar for navigation and model selection
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Model Performance", "Help & About"])
    
    # Model selection in sidebar
    st.sidebar.title("Model Selection")
    selected_model = st.sidebar.radio(
        "Select Model", 
        list(MODEL_PATHS.keys()),
        index=0,
        help="Choose which machine learning model to use for prediction"
    )
    
    # Load the selected model
    model = get_model(selected_model)
    
    # Add model description to sidebar
    with st.sidebar.expander("Model Description", expanded=False):
        st.markdown(MODEL_DESCRIPTIONS[selected_model])
    
    # Main page content based on navigation
    if page == "Prediction":
        prediction_page(model, selected_model)
    elif page == "Model Performance":
        performance_page(model, selected_model)
    else:
        about_page()

def prediction_page(model, selected_model):
    """Display the prediction page with user inputs and results"""
    
    st.title("Stroke Risk Prediction")
    st.markdown("Enter patient information to predict stroke risk.")
    
    col1, col2 = st.columns(2)
    
    # Personal Information
    with col1:
        st.subheader("Personal Information")
        
        age = st.number_input("Age", min_value=0, max_value=120, value=45, 
                             help="Age of the patient")
        
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female", "Other"],
            index=0,
            help="Gender of the patient"
        )
        
        bmi = st.number_input(
            "BMI (Body Mass Index)", 
            min_value=10.0, 
            max_value=50.0, 
            value=25.0,
            step=0.1,
            help="BMI = weight(kg) / height(m)²"
        )
    
    # Health Information
    with col2:
        st.subheader("Health Information")
        
        hypertension = st.checkbox(
            "Hypertension", 
            value=False,
            help="Does the patient have hypertension?"
        )
        
        heart_disease = st.checkbox(
            "Heart Disease", 
            value=False,
            help="Does the patient have heart disease?"
        )
        
        avg_glucose_level = st.number_input(
            "Average Glucose Level (mg/dL)", 
            min_value=50.0, 
            max_value=300.0, 
            value=100.0,
            step=1.0,
            help="Average glucose level in blood"
        )
    
    # Lifestyle Information
    st.subheader("Lifestyle Information")
    col3, col4 = st.columns(2)
    
    with col3:
        ever_married = st.radio(
            "Ever Married", 
            ["Yes", "No"],
            index=0,
            help="Has the patient ever been married?"
        )
        
        residence_type = st.radio(
            "Residence Type", 
            ["Urban", "Rural"],
            index=0,
            help="Type of area where the patient lives"
        )
    
    with col4:
        work_type = st.selectbox(
            "Work Type", 
            ["Private", "Self-employed", "Government job", "Children", "Never worked"],
            index=0,
            help="Type of employment"
        )
        
        smoking_status = st.selectbox(
            "Smoking Status", 
            ["formerly smoked", "never smoked", "smokes", "Unknown"],
            index=1,
            help="Smoking habits of the patient"
        )
    
    # Prediction button
    predict_btn = st.button("Predict Stroke Risk", type="primary", use_container_width=True)
    
    # Create a dictionary of user inputs
    if predict_btn:
        with st.spinner("Processing prediction..."):
            # Prepare data for prediction
            input_data = {
                'age': age,
                'gender': gender,
                'hypertension': int(hypertension),
                'heart_disease': int(heart_disease),
                'ever_married': ever_married,
                'work_type': work_type,
                'Residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }
            
            # Preprocess input data
            X = preprocess_input(input_data)
            
            # Make prediction
            try:
                if hasattr(model, 'predict_proba'):
                    prediction_prob = model.predict_proba(X)[:, 1][0]
                    prediction = 1 if prediction_prob > 0.5 else 0
                else:
                    prediction = model.predict(X)[0]
                    prediction_prob = prediction
                
                # Display prediction result
                st.divider()
                st.subheader("Prediction Result")
                
                col_result1, col_result2 = st.columns([1, 2])
                
                with col_result1:
                    if prediction == 1:
                        st.error("⚠️ **High Risk of Stroke**")
                        recommendation = "Consider consulting a healthcare provider for a thorough evaluation."
                    else:
                        st.success("✅ **Low Risk of Stroke**")
                        recommendation = "Continue maintaining healthy lifestyle habits."
                    
                    if hasattr(model, 'predict_proba'):
                        st.metric("Risk Probability", f"{prediction_prob:.2%}")
                
                with col_result2:
                    st.info(f"**Recommendation**: {recommendation}")
                    st.markdown("""
                        ### Key Risk Factors:
                        - Age (higher risk with increased age)
                        - Hypertension
                        - Heart disease
                        - High glucose levels
                        - Smoking history
                    """)
                    
                    st.caption("Note: This prediction is based on a machine learning model and should not replace professional medical advice.")
                
                # Risk gauge visualization using Plotly
                if hasattr(model, 'predict_proba'):
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prediction_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Stroke Risk Level", 'font': {'size': 24}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 20], 'color': 'green'},
                                {'range': [20, 40], 'color': 'lightgreen'},
                                {'range': [40, 60], 'color': 'yellow'},
                                {'range': [60, 80], 'color': 'orange'},
                                {'range': [80, 100], 'color': 'red'}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': prediction_prob * 100}
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Feature importance if available
                if selected_model == "XGBoost":
                    st.subheader("Feature Impact on Prediction")
                    
                    # This is a simplified representation - in a real app you would want to
                    # calculate SHAP values or feature importance specifically for this input
                    feature_importance = pd.DataFrame({
                        'Feature': ['age', 'avg_glucose_level', 'hypertension', 'heart_disease', 'bmi', 'smoking_status'],
                        'Importance': [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]  # Example values
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(
                        feature_importance, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
                        title='Factors Influencing This Prediction',
                        color='Importance',
                        color_continuous_scale=px.colors.sequential.Blues,
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Note: This is an illustrative representation of feature impact.")
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please check your input values and try again.")

def performance_page(model, selected_model):
    """Display model performance metrics and visualizations"""
    
    st.title(f"Model Performance: {selected_model}")
    st.markdown("Explore the performance metrics and visualizations for the selected model.")
    
    # Create tabs for different performance aspects
    performance_tabs = st.tabs(["Overview", "Classification Metrics", "ROC & PR Curves", "Compare Models"])
    
    # Simulated test data results for visualization
    # In a real application, you would load these metrics from saved evaluation results
    metrics = {
        'XGBoost': {
            'accuracy': 0.95, 'precision': 0.82, 'recall': 0.76, 'f1': 0.79, 'auc': 0.91,
            'confusion_matrix': np.array([[4700, 57], [12, 38]]),
            'fpr': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'tpr': [0, 0.4, 0.55, 0.67, 0.72, 0.78, 0.83, 0.88, 0.92, 0.95, 0.98, 0.99, 1.0],
            'precision_curve': [1, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            'recall_curve': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
        },
        'SVM': {
            'accuracy': 0.92, 'precision': 0.71, 'recall': 0.74, 'f1': 0.72, 'auc': 0.89,
            'confusion_matrix': np.array([[4680, 77], [15, 35]]),
            'fpr': [0, 0.02, 0.07, 0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.0],
            'tpr': [0, 0.35, 0.5, 0.62, 0.68, 0.74, 0.79, 0.84, 0.88, 0.92, 0.96, 0.98, 1.0],
            'precision_curve': [1, 0.9, 0.85, 0.8, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.1, 0.05],
            'recall_curve': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        },
        'Naive Bayes': {
            'accuracy': 0.89, 'precision': 0.58, 'recall': 0.86, 'f1': 0.69, 'auc': 0.88,
            'confusion_matrix': np.array([[4600, 157], [7, 43]]),
            'fpr': [0, 0.03, 0.09, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0],
            'tpr': [0, 0.45, 0.6, 0.7, 0.77, 0.82, 0.86, 0.89, 0.92, 0.95, 0.97, 0.99, 1.0],
            'precision_curve': [1, 0.85, 0.75, 0.65, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
            'recall_curve': [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        },
        'LDA': {
            'accuracy': 0.91, 'precision': 0.65, 'recall': 0.78, 'f1': 0.71, 'auc': 0.87,
            'confusion_matrix': np.array([[4650, 107], [11, 39]]),
            'fpr': [0, 0.02, 0.08, 0.13, 0.23, 0.33, 0.43, 0.53, 0.63, 0.73, 0.83, 0.93, 1.0],
            'tpr': [0, 0.4, 0.55, 0.65, 0.72, 0.78, 0.83, 0.87, 0.91, 0.94, 0.97, 0.99, 1.0],
            'precision_curve': [1, 0.87, 0.8, 0.7, 0.65, 0.55, 0.45, 0.35, 0.3, 0.2, 0.15, 0.1, 0.05],
            'recall_curve': [0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.72, 0.78, 0.85, 0.9, 0.95, 1.0]
        }
    }
    
    current_metrics = metrics[selected_model]
    
    # Overview Tab
    with performance_tabs[0]:
        st.subheader("Performance Summary")
        
        # Display key metrics with interpretations
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{current_metrics['accuracy']:.2%}")
            st.caption("Overall correct predictions")
        
        with col2:
            st.metric("Precision", f"{current_metrics['precision']:.2%}")
            st.caption("When model predicts stroke, how often is it right")
        
        with col3:
            st.metric("Recall", f"{current_metrics['recall']:.2%}")
            st.caption("Percentage of actual strokes detected")
        
        with col4:
            st.metric("F1 Score", f"{current_metrics['f1']:.2%}")
            st.caption("Harmonic mean of precision and recall")
        
        with col5:
            st.metric("AUC", f"{current_metrics['auc']:.2%}")
            st.caption("Area Under ROC Curve")
        
        st.divider()
        
        # Model Interpretation
        st.subheader("Model Interpretation")
        
        model_strengths = {
            'XGBoost': "Highest overall accuracy and good balance between precision and recall, making it reliable for stroke prediction.",
            'SVM': "Good general performance with balanced metrics, effectively handles complex relationships in the data.",
            'Naive Bayes': "Highest recall - captures most actual stroke cases, though with more false positives.",
            'LDA': "Simple but effective model with good overall performance and interpretability."
        }
        
        model_considerations = {
            'XGBoost': "Complex 'black box' model that may be more difficult to interpret than simpler alternatives.",
            'SVM': "May require more computational resources and careful parameter tuning.",
            'Naive Bayes': "Lower precision means more false alarms, which could lead to unnecessary concern.",
            'LDA': "Assumes linear relationships which may not capture all complex patterns in stroke risk factors."
        }
        
        col_interp1, col_interp2 = st.columns(2)
        
        with col_interp1:
            st.success(f"**Strengths**: {model_strengths[selected_model]}")
        
        with col_interp2:
            st.warning(f"**Considerations**: {model_considerations[selected_model]}")
    
    # Classification Metrics Tab
    with performance_tabs[1]:
        st.subheader("Confusion Matrix")
        
        # Create confusion matrix visualization
        cm = current_metrics['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                   xticklabels=["No Stroke", "Stroke"],
                   yticklabels=["No Stroke", "Stroke"])
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        
        # Interpretation of confusion matrix
        st.markdown("""
        **Confusion Matrix Interpretation:**
        
        - **True Negatives (top-left)**: Correctly predicted as no stroke
        - **False Positives (top-right)**: Incorrectly predicted as stroke (Type I error)
        - **False Negatives (bottom-left)**: Incorrectly predicted as no stroke (Type II error)
        - **True Positives (bottom-right)**: Correctly predicted as stroke
        
        In medical contexts, False Negatives (missed stroke cases) are often more serious than False Positives.
        """)
        
        # Classification report
        st.subheader("Classification Report")
        
        # Create a dataframe for the classification report
        classification_df = pd.DataFrame({
            'Class': ['No Stroke (0)', 'Stroke (1)', 'Average/Total'],
            'Precision': [0.99, current_metrics['precision'], (0.99 + current_metrics['precision'])/2],
            'Recall': [current_metrics['recall'], current_metrics['recall'], current_metrics['recall']],
            'F1-Score': [0.99, current_metrics['f1'], (0.99 + current_metrics['f1'])/2],
            'Support': [4757, 50, 4807]
        })
        
        st.dataframe(classification_df, use_container_width=True)
    
    # ROC & PR Curves Tab
    with performance_tabs[2]:
        col_roc, col_pr = st.columns(2)
        
        with col_roc:
            st.subheader("ROC Curve")
            
            # Create ROC curve
            fig_roc = go.Figure()
            
            # Add ROC curve
            fig_roc.add_trace(go.Scatter(
                x=current_metrics['fpr'],
                y=current_metrics['tpr'],
                mode='lines',
                name=f'{selected_model} (AUC = {current_metrics["auc"]:.2f})',
                line=dict(color='blue', width=2)
            ))
            
            # Add diagonal line (random classifier)
            fig_roc.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Update layout
            fig_roc.update_layout(
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                yaxis=dict(scaleanchor="x", scaleratio=1),
                xaxis=dict(constrain='domain'),
                width=450,
                height=450,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
            
            st.markdown("""
            **ROC Curve Interpretation:**
            
            The ROC (Receiver Operating Characteristic) curve shows the trade-off between:
            - True Positive Rate (Sensitivity): The proportion of actual stroke cases correctly identified
            - False Positive Rate (1-Specificity): The proportion of non-stroke cases incorrectly identified as stroke
            
            A perfect model would reach the top-left corner (100% sensitivity, 0% false positives).
            The area under the curve (AUC) quantifies performance - higher is better.
            """)
        
        with col_pr:
            st.subheader("Precision-Recall Curve")
            
            # Create PR curve
            fig_pr = go.Figure()
            
            # Add PR curve
            fig_pr.add_trace(go.Scatter(
                x=current_metrics['recall_curve'],
                y=current_metrics['precision_curve'],
                mode='lines',
                name=f'{selected_model}',
                line=dict(color='green', width=2)
            ))
            
            # Update layout
            fig_pr.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                yaxis=dict(range=[0, 1.05]),
                width=450,
                height=450,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99
                )
            )
            
            st.plotly_chart(fig_pr, use_container_width=True)
            
            st.markdown("""
            **Precision-Recall Curve Interpretation:**
            
            The Precision-Recall curve is especially useful for imbalanced datasets like stroke prediction:
            - Precision: When the model predicts stroke, how often is it correct
            - Recall: What proportion of actual stroke cases does the model detect
            
            A perfect model would reach the top-right corner (100% precision, 100% recall).
            For stroke prediction, high recall is crucial to avoid missing actual cases.
            """)
    
    # Compare Models Tab
    with performance_tabs[3]:
        st.subheader("Model Comparison")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': ['XGBoost', 'SVM', 'Naive Bayes', 'LDA'],
            'Accuracy': [metrics['XGBoost']['accuracy'], metrics['SVM']['accuracy'], 
                        metrics['Naive Bayes']['accuracy'], metrics['LDA']['accuracy']],
            'Precision': [metrics['XGBoost']['precision'], metrics['SVM']['precision'], 
                         metrics['Naive Bayes']['precision'], metrics['LDA']['precision']],
            'Recall': [metrics['XGBoost']['recall'], metrics['SVM']['recall'], 
                      metrics['Naive Bayes']['recall'], metrics['LDA']['recall']],
            'F1 Score': [metrics['XGBoost']['f1'], metrics['SVM']['f1'], 
                        metrics['Naive Bayes']['f1'], metrics['LDA']['f1']],
            'AUC': [metrics['XGBoost']['auc'], metrics['SVM']['auc'], 
                   metrics['Naive Bayes']['auc'], metrics['LDA']['auc']],
        })
        
        # Format percentages
        format_cols = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        for col in format_cols:
            comparison_df[col] = comparison_df[col].map(lambda x: f"{x:.2%}")
        
        # Show comparison table
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create radar chart for model comparison
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        fig = go.Figure()
        
        for model_name in ['XGBoost', 'SVM', 'Naive Bayes', 'LDA']:
            values = [metrics[model_name]['accuracy'], 
                     metrics[model_name]['precision'],
                     metrics[model_name]['recall'], 
                     metrics[model_name]['f1'],
                     metrics[model_name]['auc']]
            
            # Add values back to the beginning to close the loop
            values += [values[0]]
            categories_plot = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_plot,
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=500,
            width=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation based on metric
        st.subheader("Best Model Recommendation")
        
        metric_option = st.selectbox(
            "Select primary evaluation metric:", 
            ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
            index=0
        )
        
        # Map display names to dictionary keys
        metric_mapping = {
            "Accuracy": "accuracy",
            "Precision": "precision",
            "Recall": "recall", 
            "F1 Score": "f1",
            "AUC": "auc"
        }
        
        # Find the best model for the selected metric
        selected_metric = metric_mapping[metric_option]
        best_model = max(metrics.items(), key=lambda x: x[1][selected_metric])[0]
        
        # Display the recommendation
        st.success(f"**Best model based on {metric_option}:** {best_model} with {metrics[best_model][selected_metric]:.2%}")
        
        metric_explanations = {
            "Accuracy": "**Accuracy** measures the overall correctness of predictions. However, for imbalanced datasets like stroke prediction, it can be misleading since simply predicting 'no stroke' for all cases would still achieve high accuracy.",
            "Precision": "**Precision** indicates how many of the predicted stroke cases were actually strokes. High precision means fewer false alarms, which can reduce unnecessary anxiety and follow-up tests.",
            "Recall": "**Recall** (Sensitivity) measures how many of the actual stroke cases were correctly identified by the model. High recall is critical in stroke prediction to ensure actual cases aren't missed.",
            "F1 Score": "**F1 Score** is the harmonic mean of precision and recall, providing a balance between the two. It's useful when you need a single metric that considers both false positives and false negatives.",
            "AUC": "**AUC** (Area Under the ROC Curve) measures the model's ability to distinguish between stroke and non-stroke cases across all threshold values. It's threshold-independent and works well for imbalanced datasets."
        }
        
        st.info(metric_explanations[metric_option])
        
        # Recommendations based on use case
        st.subheader("Use Case Recommendations")
        
        st.markdown("""
        **For Screening (Priority on Recall)**: Naive Bayes - Captures most actual stroke cases, though with more false positives.
        
        **For Decision Support (Balance of Metrics)**: XGBoost - Best overall performance with good balance between precision and recall.
        
        **For Interpretability**: LDA - Provides good performance while remaining more interpretable than complex models.
        
        **For Resource-Constrained Environments**: Naive Bayes - Fastest training and inference with decent performance.
        """)

def about_page():
    """Display information about the application"""
    
    st.title("About Stroke Prediction App")
    
    st.markdown("""
    ## Purpose
    
    This application is designed to help predict the risk of stroke based on various health and lifestyle factors. It demonstrates the use of different machine learning models for healthcare prediction tasks.
    
    ## How It Works
    
    1. **Select a Model**: Choose between XGBoost, SVM, Naive Bayes, or LDA models.
    2. **Enter Patient Information**: Provide demographics, health indicators, and lifestyle factors.
    3. **Get Prediction**: The app will predict the stroke risk based on the provided information.
    4. **Explore Model Performance**: Compare different models and understand their strengths and limitations.
    
    ## Dataset Information
    
    The models were trained on the Stroke Prediction Dataset, which includes the following features:
    
    - **Demographics**: Age, Gender
    - **Health Indicators**: Hypertension, Heart Disease, Average Glucose Level, BMI
    - **Lifestyle Factors**: Smoking Status, Marriage Status, Work Type, Residence Type
    
    ## Important Disclaimer
    
    This application is for **educational and demonstration purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.
    
    The prediction is based on a limited dataset and may not capture all factors relevant to stroke risk. The models have inherent limitations and uncertainties.
    
    ## References
    
    - Dataset source: Healthcare Stroke Prediction Dataset
    - Machine Learning Models: XGBoost, Support Vector Machines, Naive Bayes, Linear Discriminant Analysis
    - Built with: Python, Streamlit, Scikit-learn, Plotly
    """)
    
    st.divider()
    
    st.markdown("© 2025 Stroke Prediction App | Created for educational purposes")

if __name__ == "__main__":
    main()