# Loan Risk Assessment Prediction Model

This repository contains a **machine learning model** to predict the **risk score** of loan applicants. It helps lenders evaluate potential risks and make data-driven decisions when approving loans.

---

## Dataset

The dataset contains 20,000 loan applications with the following features:

- **Demographics:** Age, Marital Status, Education Level, Employment Status
- **Financials:** Annual Income, Monthly Income, Savings/Checking Account Balances, Total Assets, Total Liabilities, Credit Score
- **Loan Details:** Loan Amount, Loan Duration, Loan Purpose, Monthly Debt Payments
- **Credit History:** Number of Open Credit Lines, Number of Credit Inquiries, Payment History, Previous Loan Defaults, Bankruptcy History, Length of Credit History
- **Derived Metrics:** Debt-to-Income Ratio, Credit Card Utilization Rate, Net Worth, Utility Bills Payment History

**Target Variable:** `RiskScore` – a numerical value representing the applicant's risk level.

---

## Preprocessing & Pipeline

The data is preprocessed using a **scikit-learn ColumnTransformer pipeline**:

- **Numerical features:** Standardized using `StandardScaler`
- **Ordinal categorical feature:** `EducationLevel` encoded with `OrdinalEncoder`
- **Nominal categorical features:** `EmploymentStatus`, `HomeOwnershipStatus`, `MaritalStatus`, `LoanPurpose` encoded with `OneHotEncoder`
- **Stratified train-test split** ensures representative risk score distribution in both sets

---

## Model

- **Algorithm:** `RandomForestRegressor`
- **Training:** Uses the processed training data from the pipeline
- **Evaluation Metrics:**
  - Root Mean Squared Error (RMSE) :  3.035
  - Mean Absolute Error (MAE) : 1.925
  - R² Score : 0.846
- **Visualizations:** Scatter plot of actual vs predicted, residual plots, and histograms comparing distributions

---
