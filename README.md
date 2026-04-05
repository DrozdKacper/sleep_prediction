# Sleep Disorder Prediction System

## Overview
Sleep disorders such as insomnia and sleep apnea are often difficult to detect early, yet they significantly impact health and quality of life.

This project presents an end-to-end machine learning system that predicts sleep disorders based on health and lifestyle data, providing users with early warnings and actionable insights.

**Notebook:** [`sleep-disorder-prediction.ipynb`](./sleep-disorder-prediction.ipynb)

---

## Project Goals
- Detect sleep disorders early (Insomnia, Sleep Apnea, None)
- Identify key risk factors affecting sleep health
- Provide a decision-support tool for preventive action
- Provide real-time predictions through a cloud-deployed machine learning API

---

## Key Features
- End-to-End ML Pipeline (data → model → deployment)
- Exploratory Data Analysis with actionable insights
- Model comparison and hyperparameter tuning
- Model interpretability (SHAP, feature importance)
- REST API for real-time predictions
- Cloud deployment of a production-ready ML system

---

## Exploratory Data Analysis (EDA)

### Dataset Overview
- 13 features (health and lifestyle data)
- Target distribution:
  - No disorder: ~58.6%
  - Sleep Apnea: ~20.9%
  - Insomnia: ~20.6%

### Key Insights
- Higher stress levels are associated with shorter sleep duration and lower sleep quality
- Increased heart rate correlates with higher stress
- Strong relationships:
  - Physical activity and daily steps
  - Systolic and diastolic blood pressure
- Sleep disorders are influenced by both physiological and lifestyle factors

---

## Data Preprocessing
- Removed non-informative features (Person ID)
- Standardized categorical variables (BMI categories)
- Feature engineering:
  - Split Blood Pressure into Systolic and Diastolic
- Missing values:
  - Target missing values interpreted as no disorder
- Scaling and encoding using Pipeline and ColumnTransformer

---

## Model Training and Evaluation

### Models Tested
- Logistic Regression
- Random Forest
- XGBoost
- SVM
- K-Nearest Neighbors

### Best Results (After Hyperparameter Tuning)

| Model | Accuracy | Macro F1 |
|------|--------|---------|
| Logistic Regression | 0.907 | 0.88 |
| XGBoost | 0.893 | 0.85 |
| Random Forest | 0.880 | 0.83 |

### Key Takeaways
- Logistic Regression achieved the best balance across all classes
- XGBoost improved after feature engineering, approaching Logistic Regression performance
- Simpler models can outperform complex ones on structured datasets

---

## Model Interpretability

### Feature Importance Insights
- Most important features:
  - Blood pressure (Systolic, Diastolic)
  - BMI category
  - Physical activity
  - Daily steps

### SHAP Insights
- High blood pressure is a strong indicator of sleep apnea
- Low activity is associated with insomnia
- Normal BMI is a strong predictor of no disorder

### Misclassification Analysis
- Errors occur due to overlapping feature patterns
- Example:
  - High blood pressure may lead to misclassification as Sleep Apnea
- Improvement idea:
  - Add domain-specific features (e.g. snoring, sleep interruptions)

---

## ML Pipeline
- Data preprocessing (imputation, scaling, encoding)
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning (GridSearchCV, CV=5)
- Final model wrapped in Pipeline

---

## API and Deployment
- Built a REST API using FastAPI for real-time predictions
- Deployed the machine learning model in the cloud
- Hosted the service using Render
- Enabled end-to-end inference pipeline from user input to prediction
