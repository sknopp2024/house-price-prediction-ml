# House Price Prediction with Machine Learning

This project explores the prediction of residential house prices using machine learning techniques on the Kaggle dataset  
**"House Prices: Advanced Regression Techniques"**.

The goal is to build a robust end-to-end machine learning pipeline that goes beyond simple model training and includes:
data exploration, feature engineering, model comparison, hyperparameter tuning, and model interpretability.

In addition to predicting house prices, the project focuses on understanding which factors drive property value and how different models capture these relationships.

---

## Problem Statement

Given structured tabular data describing residential homes (e.g., size, quality, location, construction year), the task is to predict the final sale price of each house as accurately as possible.

---

## Project Goals

- Build a complete end-to-end ML pipeline
- Explore and understand the dataset (EDA)
- Handle missing values and categorical variables
- Engineer meaningful features
- Compare multiple ML models
- Optimize model performance
- Interpret predictions using explainability methods

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP

---

## Dataset

The dataset contains 1460 training samples and 81 features, including:

- Numerical features (e.g., living area, basement size)
- Categorical features (e.g., neighborhood, building type)
- Ordinal quality ratings (e.g., kitchen quality, overall condition)

Target variable:
- `SalePrice` (log-transformed for modeling stability)

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Target distribution analysis
- Missing value investigation
- Correlation analysis
- Outlier detection

### 2. Data Preprocessing
- Train/test split
- Missing value imputation
- Scaling of numerical features
- One-hot encoding for categorical features
- Ordinal encoding for quality features

### 3. Feature Engineering
- Total area features
- Quality-based interaction features
- Aggregated structural features

### 4. Model Training
- Linear Regression
- Ridge Regression
- Random Forest
- XGBoost

### 5. Model Evaluation
- RMSE (Root Mean Squared Error)
- R² score
- Cross-validation

### 6. Model Interpretation
- Feature importance analysis
- SHAP values for global and local interpretability

---

## Results Summary

The table below summarizes **Cross-Validation (CV) RMSE** for baseline models before and after feature engineering. For Ridge Regression, Random Forest and XGBoost the optimised hyperparameters were used.


#### Performance (CV Mean RMSE)
| Model              | No FE | With FE |
|-------------------|------|--------|
| Linear Regression  | 0.1563 | 0.1531 |
| Ridge Regression   | 0.1448 | 0.1417 |
| Random Forest      | 0.1434 | 0.1375 |
| XGBoost            | 0.1242 | 0.1265 |

#### Stability (CV Std RMSE)
| Model              | No FE | With FE |
|-------------------|------|--------|
| Linear Regression  | 0.0336 | 0.0290 |
| Ridge Regression   | 0.0309 | 0.0299 |
| Random Forest      | 0.0180 | 0.0182 |
| XGBoost            | 0.0129 | 0.0108 |

### Conclusions

Feature engineering leads to consistent but model-dependent effects on cross-validation performance.

Overall, feature engineering has a **moderate and mixed impact** on model performance. While linear and tree-based models benefit to some extent, XGBoost performance slightly deteriorates, indicating that more complex models may already extract sufficient structure from the original features. The most consistent effect across all models is a small reduction in variance, suggesting improved stability across cross-validation folds.

---

## Key Insights

- Feature engineering improves model performance consistently
- Linear models perform surprisingly well on structured tabular data
- Tree-based models capture non-linear interactions better
- House price is mainly driven by:
  - Living area
  - Overall quality
  - Interaction between size and quality

---

