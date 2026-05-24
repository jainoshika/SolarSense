# SolarSense AI ☀️

Machine learning based solar power prediction and panel health monitoring system with an interactive Streamlit dashboard.

### Live Demo

**Streamlit:** https://sense-solar.streamlit.app/

**Repository:** https://github.com/jainoshika/SolarSense

---

## Features

* Solar power prediction
* GPS-based weather detection
* Manual input mode
* Multiple ML model comparison
* Panel anomaly detection
* Interactive Streamlit dashboard

---

## Workflow

```text
Raw Data
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
EDA
   ↓
Synthetic Data Generation
   ↓
Model Training
   ↓
Prediction + Anomaly Detection
   ↓
Dashboard
```

---

## Models Used

**Linear Regression**

* Baseline linear model

**Random Forest**

* Multiple decision trees with averaged predictions

**XGBoost**

* Sequential boosting of decision trees

**Isolation Forest**

* Detects abnormal panel behavior

---

## Model Performance

| Model             | MAE   | RMSE  | R²    |
| ----------------- | ----- | ----- | ----- |
| Linear Regression | 11.79 | 14.74 | 0.315 |
| Random Forest     | 1.30  | 2.66  | 0.978 |
| XGBoost           | 1.34  | 2.99  | 0.972 |

---

## Tech Stack

* Python 3.10
* Streamlit
* Scikit-learn
* XGBoost
* Pandas
* NumPy
* Open-Meteo API

---

## Setup

Clone repository:

```bash
git clone https://github.com/jainoshika/SolarSense.git

cd SolarSense
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app/dashboard.py
```

---

## Project Structure

```text
SolarSense/
│
├── app/
├── src/
├── models/
├── data/
├── notebooks/
├── requirements.txt
└── README.md
```
