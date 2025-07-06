# Stroke Risk Prediction: A Machine Learning Approach

![Stroke Prediction Banner](https://img.shields.io/badge/Healthcare-ML-blue) ![Models](https://img.shields.io/badge/Models-4-green) ![Python](https://img.shields.io/badge/Python-3.10+-yellow)

## Project Overview

This project implements multiple machine learning models to predict the risk of stroke based on patient health data. By comparing different algorithms (XGBoost, SVM, Naive Bayes, and Linear Discriminant Analysis), it provides insights into which approaches perform best for stroke risk prediction. A comprehensive Streamlit application brings these models into a user-friendly interface for exploration and prediction.

## Repository Structure

```
ml_projects/4_models/
├── healthcare-dataset-stroke-data.csv    # Source dataset
├── xgboost_stroke_prediction.ipynb       # XGBoost model notebook
├── svm_stroke_prediction.ipynb           # Support Vector Machine model notebook  
├── nb_stroke_prediction.ipynb            # Naive Bayes model notebook
├── lda_stroke_prediction.ipynb           # Linear Discriminant Analysis notebook
├── trained_models/                       # Saved trained model files
│   ├── xgboost_stroke_model.joblib
│   ├── svm_stroke_model.joblib
│   ├── naive_bayes_stroke_model.joblib
│   └── lda_stroke_model.joblib
└── stroke_prediction_app/                # Streamlit application
    ├── app.py                            # Main application code
    ├── utils.py                          # Helper functions
    ├── requirements.txt                  # Dependencies
    └── __pycache__/
```

## Jupyter Notebooks

Four separate notebooks explore the healthcare stroke dataset and implement different machine learning approaches:

1. **XGBoost Notebook (xgboost_stroke_prediction.ipynb)**
   - Gradient boosting approach with tree-based models
   - Handles imbalanced data with SMOTEENN
   - Includes feature importance analysis

2. **SVM Notebook (svm_stroke_prediction.ipynb)**
   - Support Vector Machine implementation with RBF kernel
   - Feature scaling and preprocessing
   - Permutation-based feature importance

3. **Naive Bayes Notebook (nb_stroke_prediction.ipynb)**
   - Probabilistic approach using Gaussian Naive Bayes
   - Analysis of class conditional distributions
   - Fast training with high recall performance

4. **LDA Notebook (lda_stroke_prediction.ipynb)**
   - Linear Discriminant Analysis for dimensionality reduction and classification
   - Interpretable model with linear decision boundary
   - Coefficient analysis for feature importance

Each notebook follows a consistent structure:
- Data loading and preprocessing
- Exploratory data analysis with visualizations
- Class imbalance handling using SMOTEENN
- Model training and hyperparameter settings
- Performance evaluation using multiple metrics
- Visualization of results (ROC curves, confusion matrices, etc.)
- Model saving for later deployment

## Trained Models

The trained_models directory contains saved model files generated from each notebook:

- **xgboost_stroke_model.joblib**: Ensemble-based boosting model with high overall performance
- **svm_stroke_model.joblib**: Support Vector Machine with RBF kernel
- **naive_bayes_stroke_model.joblib**: Gaussian Naive Bayes model with high recall
- **lda_stroke_model.joblib**: Linear Discriminant Analysis model with interpretability benefits

## Streamlit Application

A comprehensive Streamlit application integrates all four models into a user-friendly interface for stroke risk prediction:

### Key Features:

1. **Model Selection**
   - Users can choose between XGBoost, SVM, Naive Bayes, and LDA models
   - Each model includes a description of its strengths and weaknesses

2. **Prediction Interface**
   - Intuitive input forms for patient demographics and health metrics
   - Organized sections for personal information, health indicators, and lifestyle factors
   - Risk gauge visualization with prediction probability
   - Personalized recommendations based on prediction results

3. **Model Performance Analysis**
   - Detailed performance metrics for each model (accuracy, precision, recall, F1-score, AUC)
   - Confusion matrix visualization with interpretation
   - ROC and Precision-Recall curves
   - Comparative radar chart for all models across key metrics
   - Best model recommendation based on user-selected priority metric

4. **Educational Component**
   - Explanation of stroke risk factors
   - Interpretation guidance for model outputs
   - Descriptions of performance metrics and their significance
   - Important medical disclaimers

### Application Pages:

- **Prediction**: Enter patient data and get risk assessment
- **Model Performance**: Explore and compare model metrics and visualizations
- **Help & About**: Information on the application and dataset

## Installation and Setup

### Prerequisites
- Python 3.10+
- pip package manager

### Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd ml_projects/4_models
```

2. Install dependencies:
```bash
cd stroke_prediction_app
pip install -r requirements.txt
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Usage

### Streamlit App

1. Navigate to the application (typically http://localhost:8501)
2. Select the model you want to use from the sidebar
3. Input patient information in the provided fields
4. Click "Predict Stroke Risk" to see the prediction result
5. Explore other tabs to compare model performance

### Notebooks

To run and experiment with the notebooks:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open any of the model notebooks
3. Run the cells sequentially to reproduce the analysis
4. Modify parameters or preprocessing steps to experiment

## Model Performance Comparison

| Model      | Accuracy | Precision | Recall | F1 Score | AUC   | Training Time |
|------------|----------|-----------|--------|----------|-------|---------------|
| XGBoost    | 95%      | 82%       | 76%    | 79%      | 91%   | Fast          |
| SVM        | 92%      | 71%       | 74%    | 72%      | 89%   | Moderate      |
| Naive Bayes| 89%      | 58%       | 86%    | 69%      | 88%   | Very Fast     |
| LDA        | 91%      | 65%       | 78%    | 71%      | 87%   | Very Fast     |

### Key Findings:

- **XGBoost** provides the best overall performance with balanced precision and recall
- **Naive Bayes** excels at detecting potential stroke cases (highest recall) but with more false positives
- **LDA** offers a good balance of performance and interpretability
- **SVM** shows strong general performance with good AUC

## Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Imbalanced-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Application**: Streamlit
- **Model Persistence**: Joblib

## Future Improvements

- Implement more advanced models (Neural Networks, Random Forests)
- Add feature engineering capabilities within the app
- Incorporate more sophisticated SHAP value visualizations for model interpretability
- Implement time-series analysis for patients with multiple measurements
- Add confidence intervals to predictions
- Develop API endpoint for integration with other systems

## Important Disclaimer

This application is for **educational and demonstration purposes only**. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical concerns.

The predictions are based on a limited dataset and may not capture all factors relevant to stroke risk. The models have inherent limitations and uncertainties that must be considered in any real-world application.

## References

- Dataset: Healthcare Stroke Prediction Dataset
- Imbalanced learning techniques: SMOTEENN from imbalanced-learn
- Model implementations: Scikit-learn and XGBoost documentation
- Visualization approaches: Plotly and Seaborn galleries

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the open-source community for the tools and libraries that made this project possible
- Special thanks to healthcare professionals who provided domain knowledge guidance

---

*Developed for educational purposes in healthcare predictive modeling*