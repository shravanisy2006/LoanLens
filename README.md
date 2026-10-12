# LoanLens — Loan Default Prediction

LoanLens is a machine learning web application that estimates the probability of loan default using borrower and loan-related information from the LendingClub dataset.

## Live Demo

[Launch LoanLens](https://loanlens-moadgx936dctbmhimtlc2h.streamlit.app/)

## Features

- Interactive loan default prediction interface
- Exploratory data analysis and visualizations
- Comparison of Logistic Regression, Decision Tree, and Random Forest
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Interactive loan scenario exploration
- Streamlit web interface

## Tech Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Plotly
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## Machine Learning Workflow

1. Load and inspect the LendingClub dataset.
2. Clean the data and handle missing values.
3. Select features available at loan origination to reduce data leakage.
4. Encode categorical variables and scale numerical features.
5. Train and compare multiple classification models.
6. Evaluate performance using cross-validation and test-set metrics.
7. Deploy the trained model through Streamlit.

## Run Locally

Clone the repository and navigate to the project directory.

```bash
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

## Project Structure

```text
LoanLens/
├── assets/
├── data/
├── models/
├── src/
│   └── eda.py
├── app.py
├── train_model.py
└── requirements.txt
```

## Important Note

LoanLens is an educational machine learning project. Its predictions are estimates based on historical data and should not be treated as financial advice or used as the sole basis for lending decisions.

## Author

Developed as a machine learning project focused on data analysis, classification, model evaluation, and web application deployment.
