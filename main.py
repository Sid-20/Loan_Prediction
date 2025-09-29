import os
import pandas as pd
import numpy as np
import joblib
 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(numerical_col,ordinal_categorical_col,onehot_categorical_col):

    ord_values=[['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']]

    num_pipeline=Pipeline([
        ("scaler",StandardScaler())
    ])

    ordinal_cat_pipeline=Pipeline([
        ("ordinal",OrdinalEncoder( categories=ord_values ))
    ])

    one_hot_pipeline=Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, numerical_col),
        ("ordinal", ordinal_cat_pipeline, ordinal_categorical_col),
        ("onehot", one_hot_pipeline, onehot_categorical_col)
    ])
    return full_pipeline



if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    data=pd.read_csv('Loan.csv')
    data.drop(columns=['InterestRate','LoanApproved','BaseInterestRate','MonthlyLoanPayment','TotalDebtToIncomeRatio','ApplicationDate'],axis=1,inplace=True)

    data['RiskScore_bin'] = pd.qcut(data['RiskScore'], q=10, labels=False)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in split.split(data, data['RiskScore_bin']):
        train_data = data.iloc[train_idx].copy()
        #test_data = data.iloc[test_idx].copy()
    
    train_data.drop('RiskScore_bin', axis=1, inplace=True)

    #trained_data=train_data.copy()

    prepared_data=train_data.drop(columns=['RiskScore'],axis=1)

    labeled_data=train_data['RiskScore'].copy()

    numerical_col=['Age','AnnualIncome','CreditScore','Experience','LoanAmount','LoanDuration','NumberOfDependents',
                   'MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines','NumberOfCreditInquiries',
                   'DebtToIncomeRatio','BankruptcyHistory','PreviousLoanDefaults','PaymentHistory','LengthOfCreditHistory',
                   'SavingsAccountBalance','CheckingAccountBalance','TotalAssets','TotalLiabilities','MonthlyIncome',
                   'UtilityBillsPaymentHistory','JobTenure','NetWorth']
    ordinal_categorical_col=['EducationLevel']
    onehot_categorical_col=['EmploymentStatus', 'HomeOwnershipStatus', 'MaritalStatus', 'LoanPurpose']

    pipeline = build_pipeline(numerical_col,ordinal_categorical_col,onehot_categorical_col)

    final_prepared_data = pipeline.fit_transform(prepared_data)
 
    model = RandomForestRegressor(random_state=42)
    model.fit(final_prepared_data, labeled_data)
 
    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
 
    print("Model trained and saved.")
    



 
else:
    # INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
 
    input_data = pd.read_csv("input_test_data.csv")
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["predicted_risk_score"] = predictions
 
    input_data.to_csv("output_predicted.csv", index=False)
    print("Inference complete. Results saved to output.csv")