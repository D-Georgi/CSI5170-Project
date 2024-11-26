import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

# Load datasets
df_no_missing = pd.read_csv('data/blood_glucose_30min_avg.csv')
df_missing = pd.read_csv('data/blood_glucose_30min_avg_keep_missing.csv')

# Inspect the datasets
print("Dataset without missing values:")
print(df_no_missing.info())
print("\nDataset with missing values:")
print(df_missing.info())


target_column = 'bg+1:00'

def train_evaluate_ridge(df, target_column, alpha=1.0, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    identifier_cols = ['id', 'p_num', 'time']
    for col in identifier_cols:
        if col in X.columns:
            X = X.drop(columns=[col])
            if col in numeric_features:
                numeric_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create a pipeline with preprocessing and Ridge Regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('ridge', Ridge(alpha=alpha))
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Predict on the test data
    y_pred = pipeline.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'MSE': mse, 'R2': r2, 'Model': pipeline, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}

# Train and evaluate models
results_no_missing = train_evaluate_ridge(df_no_missing, target_column, alpha=1.0)
results_missing = train_evaluate_ridge(df_missing, target_column, alpha=1.0)

# Print performance
print("Performance on Dataset without Missing Values:")
print(f"Mean Squared Error (MSE): {results_no_missing['MSE']:.4f}")
print(f"R-squared (R²): {results_no_missing['R2']:.4f}\n")

print("Performance on Dataset with Missing Values:")
print(f"Mean Squared Error (MSE): {results_missing['MSE']:.4f}")
print(f"R-squared (R²): {results_missing['R2']:.4f}")