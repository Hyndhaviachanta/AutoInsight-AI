
# AutoInsight AI

## Explainable Dataset Intelligence Platform

### Project Overview

AutoInsight AI is a machine learning and data analytics platform developed to automate dataset profiling, data quality assessment, anomaly detection, model benchmarking, explainable AI analysis, and research report generation.

The objective of this project is to help researchers, students, and data scientists quickly evaluate datasets and gain meaningful insights before building machine learning models.

---

## Problem Statement

Real-world datasets frequently contain missing values, duplicate records, outliers, and other quality issues that can negatively affect machine learning performance. Data preprocessing and exploratory analysis often require significant time and effort.

This project addresses that challenge by providing an integrated platform that automates dataset evaluation and presents results through visualizations, benchmarking, and explainability techniques.

---

## Key Features

### User Management

* User registration and login
* Multiple user accounts
* User-specific report history

### Dataset Profiling

* Dataset shape analysis
* Missing value analysis
* Duplicate record detection
* Data type inspection
* Dataset quality scoring

### Data Visualization

* Missing value charts
* Feature distributions
* Correlation heatmaps

### Outlier Detection

* Isolation Forest-based anomaly detection
* Outlier percentage calculation
* Outlier visualization

### Dataset Reliability Assessment

* Missing data score
* Duplicate score
* Outlier score
* Overall reliability index
* Dataset grading system

### Machine Learning Benchmarking

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

### Explainable AI

* SHAP-based feature importance analysis
* Model interpretability
* Feature contribution visualization

### Automated Reporting

* Research-style PDF report generation
* Dataset summary
* Reliability metrics
* Model evaluation results
* Explainability findings

### Dataset Comparison

* Comparison of multiple datasets
* Quality metric comparison
* Structural comparison

### Administrative Dashboard

* User activity monitoring
* Report generation statistics

---

## System Architecture

Frontend:

* Streamlit
* Plotly
* HTML
* CSS

Backend:

* Python
* Pandas
* NumPy

Machine Learning:

* Scikit-learn
* Isolation Forest
* Random Forest
* Logistic Regression
* Decision Tree
* Gradient Boosting

Explainability:

* SHAP

Reporting:

* ReportLab

Storage:

* CSV-based user database
* User report repository

---

## Project Structure

```text
AutoInsight-AI/
│
├── app.py
├── users.csv
├── requirements.txt
├── README.md
│
├── reports/
├── screenshots/
└── sample_data/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Hyndhaviachanta/AutoInsight-AI.git
```

Move into the project directory:

```bash
cd AutoInsight-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the application using:

```bash
streamlit run app.py
```

---

## Workflow

1. Login or create an account
2. Upload a CSV dataset
3. Analyze dataset quality
4. Detect anomalies and outliers
5. Benchmark machine learning models
6. Generate SHAP explanations
7. Download the research report

---

## Results

The Titanic dataset was used to evaluate the platform.

Dataset Characteristics:

* Rows: 891
* Columns: 12
* Missing Values: 866
* Quality Score: 91.9%

The platform successfully generated:

* Data quality assessments
* Outlier detection results
* Model benchmarking reports
* SHAP explainability outputs
* Automated PDF reports

---

## Limitations

Current limitations include:

* Benchmarking is primarily focused on classification tasks
* User credentials are stored in a CSV file
* Dataset assistant uses rule-based responses
* Performance optimization for large datasets is limited

---

## Future Work

Potential future improvements include:

* FastAPI backend integration
* PostgreSQL database support
* Docker deployment
* Cloud hosting
* Large Language Model integration
* Automated feature engineering
* AutoML support
* Advanced anomaly detection methods

---

## References

1. Streamlit Documentation
2. Scikit-learn Documentation
3. SHAP Documentation
4. Plotly Documentation
5. ReportLab Documentation
6. Isolation Forest Research Paper
7. Titanic Dataset Documentation

---

## Author

Hyndhavi Preetham

Master of Science in Computer Science

Texas State University
