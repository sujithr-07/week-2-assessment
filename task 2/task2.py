import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#1. CREATE SAMPLE DATASET

data = {
    'age': [25, 32, 45, 28, 50, 35, 40, np.nan, 29, 55,
            38, 42, 31, 48, 27, 36, 44, 52, 33, 39],

    'income': [30000, 45000, 80000, 35000, 90000,
               60000, 70000, 40000, 32000, 100000,
               55000, 75000, 42000, 85000, 31000,
               58000, 78000, 95000, 47000, 65000],

    'loan_amount': [10000, 15000, 30000, 12000, 35000,
                    20000, 25000, 14000, 11000, 40000,
                    18000, 28000, 13000, 32000, 10000,
                    19000, 27000, 38000, 16000, 22000],

    'employment': ['Salaried', 'Self-employed', 'Salaried',
                   'Unemployed', 'Salaried', 'Self-employed',
                   'Salaried', 'Salaried', 'Unemployed',
                   'Salaried', 'Self-employed', 'Salaried',
                   'Salaried', 'Self-employed', 'Unemployed',
                   'Salaried', 'Salaried', 'Self-employed',
                   'Salaried', 'Self-employed'],

    'education': ['Graduate', 'Postgraduate', 'Graduate',
                  'High School', 'Postgraduate', 'Graduate',
                  'Postgraduate', 'Graduate', 'High School',
                  'Postgraduate', 'Graduate', 'Postgraduate',
                  'Graduate', 'Postgraduate', 'High School',
                  'Graduate', 'Postgraduate', 'Graduate',
                  'Graduate', 'Postgraduate'],

    'application_date': pd.date_range(
        start='2024-01-01', periods=20, freq='30D'
    ),

    'loan_status': [0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
                    1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

#2. FEATURE ENGINEERING

# Date decomposition
df['application_year'] = df['application_date'].dt.year
df['application_month'] = df['application_date'].dt.month

# Interaction feature
df['income_loan_ratio'] = (
    df['income'] / df['loan_amount']
)

# Log transformation of skewed loan amount
df['log_loan_amount'] = np.log1p(df['loan_amount'])

# Drop original date column
df.drop(columns=['application_date'], inplace=True)

#3. SEPARATE FEATURES AND TARGET

X = df.drop(columns=['loan_status'])
y = df['loan_status']

#4. DEFINE COLUMN TYPES

numeric_features = [
    'age',
    'income',
    'loan_amount',
    'application_year',
    'application_month',
    'income_loan_ratio',
    'log_loan_amount'
]

categorical_features = ['employment']

ordinal_features = ['education']

education_order = [
    ['High School', 'Graduate', 'Postgraduate']
]

#5. NUMERIC PIPELINE - MEDIAN IMPUTATION

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

#6. CATEGORICAL PIPELINE

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

#7. ORDINAL PIPELINE

ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=education_order,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

#8. COLUMN TRANSFORMER

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features),
    ('ord', ordinal_pipeline, ordinal_features)
])

#9. COMPLETE MACHINE LEARNING PIPELINE

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

#10. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#11. CROSS VALIDATION

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print("Cross Validation Scores:", cv_scores)
print("Pipeline CV accuracy:", round(cv_scores.mean(), 2))

#12. TRAIN FINAL MODEL

model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))

#13. SAVE FITTED PIPELINE

joblib.dump(model, 'loan_approval_pipeline.pkl')

print("Pipeline saved successfully!")

#14. LOAD FITTED PIPELINE

loaded_model = joblib.load('loan_approval_pipeline.pkl')

#15. PREDICT 3 NEW RAW ROWS

new_data = pd.DataFrame({
    'age': [30, 45, 26],
    'income': [50000, 85000, 28000],
    'loan_amount': [15000, 30000, 12000],
    'employment': [
        'Salaried',
        'Self-employed',
        'Unemployed'
    ],
    'education': [
        'Graduate',
        'Postgraduate',
        'High School'
    ],
    'application_year': [2025, 2025, 2025],
    'application_month': [3, 6, 9],
    'income_loan_ratio': [
        50000 / 15000,
        85000 / 30000,
        28000 / 12000
    ],
    'log_loan_amount': np.log1p(
        [15000, 30000, 12000]
    )
})

predictions = loaded_model.predict(new_data)

print("Predictions for 3 new rows:", predictions)
