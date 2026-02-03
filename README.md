# Heart Disease Prediction using Machine Learning

[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

## Overview

Preventing heart diseases is a critical global health priority. Data-driven systems for predicting heart diseases can improve research and prevention, helping more people live healthy lives. This project leverages Machine Learning to predict the presence of heart disease with high accuracy.

The project involves comprehensive analysis of the heart disease patient dataset with advanced data processing, feature engineering, model optimization, and ensemble methods to achieve superior predictive performance.

## Dataset Information

- **Dataset Size**: 303 samples with 14 features
- **Features**: age, sex, chest pain type (cp), resting blood pressure (trestbps), cholesterol (chol), fasting blood sugar (fbs), resting electrocardiographic results (restecg), maximum heart rate achieved (thalach), exercise induced angina (exang), oldpeak, slope, number of major vessels (ca), thalassemia (thal)
- **Target Variable**: Binary classification (0 = no disease, 1 = disease present)
- **Class Distribution**: 165 patients with heart disease, 138 without
- **Source**: UCI Machine Learning Repository - Cleveland Heart Disease Dataset

## Machine Learning Algorithms Implemented

1. **Logistic Regression** (with feature scaling)
2. **K-Nearest Neighbours** (with feature scaling)
3. **Support Vector Machine** (RBF/Linear kernels with hyperparameter tuning)
4. **Decision Tree** (with pruning and hyperparameter optimization)
5. **Random Forest** (ensemble method with GridSearchCV optimization)
6. **XGBoost** (gradient boosting with hyperparameter tuning)
7. **Neural Network** (Deep Learning with Keras/TensorFlow)
8. **Voting Classifier** (ensemble of top-performing models)

## Key Features & Improvements

- **Robust Preprocessing**: Proper One-Hot Encoding for categorical features and Standard Scaling for numerical features
- **Scikit-learn Pipelines**: Clean code organization preventing data leakage during training and testing
- **Hyperparameter Tuning**: Systematic optimization using `GridSearchCV` with Cross-Validation
- **Ensemble Methods**: Voting Classifier combining multiple top-performing models for enhanced accuracy
- **Comprehensive Evaluation**: Confusion Matrices, Precision-Recall curves, ROC-AUC scores, and Cross-Validation metrics
- **Feature Importance Analysis**: Identification of most predictive attributes

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/[USERNAME]/Heart-Disease-Prediction-using-Machine-Learning.git
   cd Heart-Disease-Prediction-using-Machine-Learning
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter tensorflow xgboost
   ```

3. Ensure you have the following packages installed:
   - pandas >= 1.3.0
   - numpy >= 1.21.0
   - scikit-learn >= 1.0.0
   - matplotlib >= 3.4.0
   - seaborn >= 0.11.0
   - jupyter >= 1.0.0
   - tensorflow >= 2.6.0
   - xgboost >= 1.5.0

## Usage

1. Make sure you have Jupyter Notebook installed or use Google Colab
2. The main notebook is `Heart_disease_prediction.ipynb` for the primary implementation
3. An optimized version is available as `Heart_disease_prediction_optimized.ipynb` with enhanced preprocessing and model optimization
4. A deep learning approach is demonstrated in `Heart Diseases Prediction With Deep Learning.ipynb`
5. Run the notebooks cell by cell to reproduce the analysis and predictions

## Expected Performance

The optimized models achieve:
- Accuracy: ~85-90%
- Precision: ~85-90%
- Recall: ~85-90%
- F1-Score: ~85-90%
- AUC-ROC: ~0.90+

Actual results may vary depending on train/test splits and hyperparameter configurations.

## File Structure

```
├── Heart Diseases Prediction With Deep Learning.ipynb    # Deep learning implementation
├── Heart_disease_prediction.ipynb                      # Main ML implementation
├── Heart_disease_prediction_optimized.ipynb            # Optimized version with best practices
├── heart.csv                                           # Dataset file
└── README.md                                           # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details (if available in the repository).

## Acknowledgments

- The dataset is sourced from the UCI Machine Learning Repository
- Inspired by various heart disease prediction research papers
- Built with scikit-learn, TensorFlow, and other open-source libraries