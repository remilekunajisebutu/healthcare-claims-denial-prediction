# Claims Denial Prediction Project

A GitHub-ready Python project that predicts whether a healthcare claim is likely to be denied.

## Project Overview
This project simulates a realistic healthcare revenue cycle workflow:
- Generate or use claims data
- Clean and prepare the dataset
- Train a machine learning model
- Evaluate model performance
- Score new claims for denial risk
- Export results for dashboarding or business review

## Business Problem
Denied claims delay reimbursement, increase rework, and hurt cash flow.  
The goal of this project is to identify high-risk claims early so billing or revenue cycle teams can review them before submission.

## Target Variable
- `denial_flag`
  - `1` = denied claim
  - `0` = paid/approved claim

## Features Included
The synthetic dataset includes variables commonly associated with claim denials:
- payer type
- claim amount
- service type
- prior authorization status
- diagnosis validity
- coding accuracy
- claim submission lag
- provider specialty
- patient age
- place of service
- coverage status
- duplicate claim indicator
- missing modifier indicator
- insurance verification status

## Project Structure
```bash
claims_denial_prediction_project/
│
├── data/
│   ├── raw/
│   │   └── claims_denial_synthetic.csv
│   └── processed/
│
├── models/
│
├── outputs/
│
├── notebooks/
│
├── src/
│   ├── generate_synthetic_data.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate synthetic data
```bash
python src/generate_synthetic_data.py
```

### 4. Train the model
```bash
python src/train_model.py
```

### 5. Score claims
```bash
python src/predict.py
```

## Outputs
After training, the project creates:
- `models/denial_model.pkl`
- `outputs/model_metrics.json`
- `outputs/feature_importance.csv`
- `outputs/scored_claims.csv`

## Suggested GitHub Repo Name
- `claims-denial-prediction`
- `healthcare-claims-denial-ml-project`
- `revenue-cycle-denial-prediction`

## Recommended GitHub README Add-ons
When you publish this to GitHub, you can add:
- a screenshot of model outputs
- a confusion matrix image
- business recommendations
- future improvements such as XGBoost, SHAP explainability, or dashboard integration

## Future Improvements
- Add hyperparameter tuning
- Use real claim data if available
- Build a Streamlit app
- Connect scored outputs to Power BI
- Add model explainability using SHAP
