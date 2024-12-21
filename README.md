# DataBeez_Test_Tech_DataScience

## Credit Scoring Machine Learning Project

## Overview
This project focuses on building a machine learning model for credit scoring, which assesses the creditworthiness of individuals. The model predicts whether a loan applicant is likely to default based on historical data and various financial and demographic features. Credit scoring is a critical task for financial institutions to minimize risks and improve decision-making.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Tools and Technologies](#tools-and-technologies)
- [Installation](#installation)
- [Run](#run)

---

## Project Structure
```
├── data/                  # Credit Scoring dataset
├── CVs/                   # Resumes of team members
├── models.pkl             # Saved model 
├── app.py                 # The credit_scoring app
├── requirements.txt       # Dependencies
├── processing.pkl         # Saved preprocessing pipeline
├── credit_scoring.ipynb   # Project Jupyter notebook
├── README.md              # Project documentation
└── Dockerfile             # Dockerfile
```

---

## Dataset

### Source
The dataset used for this project was sourced from Kaggle. It contains detailed information about individuals, including demographic, financial, and loan application data.

### Features
- `Age` (numeric): Applicant's age.
- `Sex` (categorical): Male or Female.
- `Job` (numeric):
  - 0: Unskilled and non-resident
  - 1: Unskilled and resident
  - 2: Skilled
  - 3: Highly skilled
- `Housing` (categorical): Own, Rent, or Free.
- `Saving accounts` (categorical): Little, Moderate, Quite Rich, Rich.
- `Checking account` (categorical): Little, Moderate, Rich.
- `Credit amount` (numeric): Amount in DM (Deutsche Mark).
- `Duration` (numeric): Loan duration in months.
- `Purpose` (categorical): Car, Furniture/Equipment, Radio/TV, Domestic Appliances, Repairs, Education, Business, Vacation/Others.
- `Risk` (binary target): Good or Bad Risk.

### Preprocessing
The dataset underwent the following preprocessing steps:
- Handling missing values.
- Encoding categorical features.
- Normalizing numerical features.
- Splitting into training, validation, and test sets.

---

## Modeling Approach

### Steps
1. **Exploratory Data Analysis (EDA):**
   - Visualized data distributions, correlations, and outliers.
   - Identified key features affecting creditworthiness.

2. **Feature Engineering:**
   - Derived new features to improve model performance.

3. **Model Selection:**
   - Evaluated models such as Logistic Regression, Random Forest, Gradient Boosting, and AdaBoost.
   - Tuned hyperparameters using Grid Search and Random Search.

4. **Evaluation:**
   - Assessed models using metrics such as Accuracy, Precision, Recall, and F1-score.

---

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:**
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Modeling: `scikit-learn`
  - Deployment: `Streamlit`, `Docker`
- **Environment:** Jupyter Notebook, VSCode, Docker

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aime-Rick/Hack2Hire_TestTech_DataScience_1.git
   cd Hack2Hire_TestTech_DataScience_1
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run

### Local

1. Start the application:
   ```bash
   python -m streamlit run predictions.py
   ```

2. Open a web browser and navigate to:
   [http://localhost:8501](http://localhost:8501)

### Docker

1. Build the Docker image:
   ```bash
   docker build -t credit_scoring_app .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -it credit_scoring_app
   ```

3. Open a web browser and navigate to:
   [http://localhost:8501](http://localhost:8501)

