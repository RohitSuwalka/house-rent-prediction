# 🏠 House Rent Prediction

An end-to-end machine learning project that predicts monthly house rent across Indian cities using a **Gradient Boosting Regressor** backend (Flask REST API) and an interactive **Streamlit** frontend.

---

## Project Structure

```
house_rent_prediction/
├── train_model.py        # Data preprocessing + model training
├── app.py                # Flask REST API (backend)
├── streamlit_app.py      # Streamlit web app (frontend)
├── requirements.txt      # Python dependencies
├── model/
│   ├── rent_model.pkl    # Trained GradientBoostingRegressor (auto-generated)
│   └── encoders.pkl      # LabelEncoders + feature metadata (auto-generated)
└── README.md

House_Rent_Dataset.csv    # Raw dataset (project root)
```

---

## Prerequisites

- Python **3.9 or later**
- pip

---

## Installation

```bash
# 1. Clone / download the project, then navigate to the workspace root
cd <project-root>          # folder that contains House_Rent_Dataset.csv

# 2. Install dependencies
pip install -r house_rent_prediction/requirements.txt
```

---

## Quick Start (3 steps)

### Step 1 — Train the model

```bash
python house_rent_prediction/train_model.py
```

Expected output:
```
Loading dataset ...
  Shape: (4746, 12)
Training Gradient Boosting Regressor ...

-- Model Evaluation --
  MAE  : Rs.13,814
  RMSE : Rs.48,417
  R2   : 0.4118
----------------------

Model    saved -> .../model/rent_model.pkl
Encoders saved -> .../model/encoders.pkl
```

> **Must be run first** — the API and the Streamlit app both need the saved model files.

---

### Step 2 — Start the Flask API  *(optional — Streamlit works standalone)*

Open a **new terminal** and run:

```bash
python house_rent_prediction/app.py
```

The API starts at `http://127.0.0.1:5000`.

| Method | Endpoint               | Description                            |
|--------|------------------------|----------------------------------------|
| GET    | `/health`              | Service health check                   |
| GET    | `/options`             | All valid dropdown values              |
| POST   | `/predict`             | Predict rent from JSON input           |
| GET    | `/feature_importance`  | Feature importance scores              |

**Example — predict via curl:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{\"BHK\":2,\"Size\":900,\"Bathroom\":2,\"floor\":\"2 out of 5\",\"Area Type\":\"Super Area\",\"City\":\"Mumbai\",\"Furnishing Status\":\"Semi-Furnished\",\"Tenant Preferred\":\"Bachelors/Family\",\"Point of Contact\":\"Contact Owner\"}"
```

**Response:**
```json
{
  "currency": "INR",
  "formatted": "Rs. 43,653",
  "predicted_rent": 43652.73
}
```

---

### Step 3 — Launch the Streamlit frontend

Open another terminal and run:

```bash
python -m streamlit run house_rent_prediction/streamlit_app.py
```

Open your browser at **http://localhost:8501**

> The Streamlit app loads the model **directly** — Flask does not need to be running for it to work.

---

## Features

### Prediction Tab
- Live rent estimate updated as you change sidebar inputs
- Property summary table
- Comparable listings pulled from the dataset
- Rent distribution histogram with your estimate marked

### Data Insights Tab
- KPI cards: total listings, average rent, median rent, cities covered
- Median rent by city (bar chart)
- Median rent by furnishing status (bar chart)
- Size vs Rent scatter plot (1,000-sample)
- BHK distribution (donut chart)
- Rent by BHK across cities (box plot)

### Feature Importance Tab
- Horizontal bar chart of Gradient Boosting feature importances
- Model configuration summary

---

## Input Fields

| Field              | Type    | Example values                                      |
|--------------------|---------|-----------------------------------------------------|
| City               | Select  | Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad |
| BHK                | Slider  | 1 – 6                                               |
| Size               | Number  | 100 – 10,000 sq ft                                  |
| Bathrooms          | Slider  | 1 – 6                                               |
| Current Floor      | Number  | 0 (Ground) – 50                                     |
| Total Floors       | Number  | 1 – 50                                              |
| Area Type          | Select  | Super Area, Carpet Area, Built Area                 |
| Furnishing Status  | Select  | Furnished, Semi-Furnished, Unfurnished              |
| Tenant Preferred   | Select  | Bachelors, Family, Bachelors/Family                 |
| Point of Contact   | Select  | Contact Owner, Contact Agent, Contact Builder       |

---

## Dataset

**File:** `House_Rent_Dataset.csv`  
**Records:** 4,746 rental listings  
**Source:** Indian rental market data

| Column            | Description                        |
|-------------------|------------------------------------|
| Posted On         | Listing date                       |
| BHK               | Number of bedrooms                 |
| Rent              | Monthly rent in INR (target)       |
| Size              | Area in square feet                |
| Floor             | Floor info (e.g. "2 out of 5")     |
| Area Type         | Super Area / Carpet Area / Built   |
| Area Locality     | Neighbourhood name                 |
| City              | City name                          |
| Furnishing Status | Furnished / Semi / Unfurnished     |
| Tenant Preferred  | Preferred tenant type              |
| Bathroom          | Number of bathrooms                |
| Point of Contact  | Owner / Agent / Builder            |

---

## Model Details

| Parameter       | Value                        |
|-----------------|------------------------------|
| Algorithm       | GradientBoostingRegressor    |
| n_estimators    | 300                          |
| learning_rate   | 0.08                         |
| max_depth       | 5                            |
| subsample       | 0.8                          |
| random_state    | 42                           |
| Train/Test split| 80% / 20%                    |
| MAE             | Rs. 13,814                   |
| RMSE            | Rs. 48,417                   |
| R² Score        | 0.41                         |

---

## Tech Stack

| Layer      | Technology                                      |
|------------|-------------------------------------------------|
| ML         | scikit-learn, pandas, numpy, joblib             |
| Backend    | Flask 3.x                                       |
| Frontend   | Streamlit 1.x, Plotly                           |
| Charts     | Plotly Express, Plotly Graph Objects            |
