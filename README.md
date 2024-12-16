# DataBeez_Test_Tech_DataScience

## Credit Scoring Machine Learning Project

## Overview
This project focuses on building a machine learning model for credit scoring, which assesses the creditworthiness of individuals. The model aims to predict whether a loan applicant is likely to default based on historical data and various financial and demographic features. Credit scoring is a critical task for financial institutions to minimize risks and improve decision-making.

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
├── CVs/                   # Our CVS
├── models.pkl             # Saved model 
├── app.py                 # The credit_scoring app
├── requirements.txt       # Dependencies
├── processing.pkl         # Saved preprocessing pipeline
├── credit_scoring.ipynb   # Project jupyter notebook
├── README.md              # Project documentation
└── Dockerfile             # Dockerfile
```

---

## Dataset

### Source
The dataset used for this project was sourced from Kaggle. It contains detailed information about individuals, including demographic, financial, and loan application data.

### Features:
- `Age` (numeric)
- `Sex`(text: male, female)
- `Job` (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
- `Housing` (text: own, rent, or free)
- `Saving accounts` (text - little, moderate, quite rich, rich)
- `Checking account` (text - little, moderate, rich)
- `Credit amount` (numeric, in DM)
- `Duration` (numeric, in month)
- `Purpose`(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
- `Risk` (Value target - Good or Bad Risk)

### Preprocessing
The dataset underwent the following preprocessing steps:
- Handling missing values
- Encoding categorical features
- Normalizing numerical features
- Splitting into training, validation, and test sets

---

## Modeling Approach

### Steps
1. **Exploratory Data Analysis (EDA):**
   - Visualized data distributions, correlations, and outliers.
   - Identified key features affecting creditworthiness.

2. **Model Selection:**
   - Compared models such as Logistic Regression, Random Forest, Gradient Boosting, and AdaBoost.
   - Performed hyperparameter tuning using Grid Search/Random Search.

3. **Evaluation:**
   - Evaluated models using metrics such as accuracy, precision, recall, F1-score.

---

## Tools and Technologies
- **Programming Language:** Python
- **Libraries:**
  - Data Processing: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Modeling: scikit-learn
  - Model Deployment: Streamlit, Docker
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

## Run:
### Local

   ```bash
   python -m streamlit run app.py
   ```

Open a web browser and navigate to [http://localhost:8501]
### Docker
1. Build the docker image:
   ```bash
   docker build -t <your_image_name> .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -it <your_image_name> 
   ```
   Open a web browser and navigate to [http://localhost:8501]



