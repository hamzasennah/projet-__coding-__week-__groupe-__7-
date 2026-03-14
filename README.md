# projet-__coding-__week-__groupe-__7-# Medical Decision Support Application

## Obesity Risk Estimation with Explainable Machine Learning (SHAP)

### Project Overview

This project aims to develop a clinical decision-support system that helps physicians estimate the obesity risk of patients based on lifestyle habits and physical conditions.

The system uses machine learning models combined with SHAP explainability to provide transparent and interpretable predictions.

---

# Dataset

Dataset used:
UCI Machine Learning Repository – *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*

The dataset contains patient information such as:

* Age
* Gender
* Height and weight
* Eating habits
* Physical activity level
* Transportation usage

The target variable represents **7 obesity levels**.

---

# Project Structure

```
project/
│
├── app/
│   └── app.py
│
├── data/
│
├── models/
│
├── notebooks/
│   ├── eda.ipynb
│   └── shap_beta.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── tests/
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/your-repository/project.git
cd project
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training the Model

To train the machine learning model:

```
python src/train_model.py
```

The trained model will be saved in the **models/** folder.

---

# Running the Application

To launch the medical decision interface:

```
streamlit run app/app.py
```

Then open the browser:

```
http://localhost:8501
```

---

# Machine Learning Models Evaluated

The following models were evaluated:

* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

Performance metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The best performing model was selected based on these metrics.

---

# SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the model predictions.

The following visualizations were generated:

* SHAP summary plot
* Feature importance
* Individual prediction explanation

These visualizations help physicians understand which factors influence obesity risk.

---

# Memory Optimization

A function **optimize_memory(df)** was implemented in:

```
src/data_processing.py
```

This function reduces memory usage by converting data types:

* float64 → float32
* int64 → int32

Memory usage is compared before and after optimization.

---

# Automated Testing

The project includes automated tests in the **tests/** folder.

Examples:

* Missing values verification
* Memory optimization function
* Model loading and prediction

Tests are executed automatically using **GitHub Actions CI/CD**.

---

# Prompt Engineering

Prompt engineering was used during the development process to assist with:

* Data preprocessing
* Memory optimization
* Model evaluation

The prompts and results are documented in the project.

---

# Reproducibility

The project is fully reproducible.
A user can run the entire project using:

```
pip install -r requirements.txt
python src/train_model.py
streamlit run app/app.py
```

---

# Authors

Team 7 – Coding Week Project
