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
df_condensed = pd.read_csv('data/blood_glucose_30min_avg_keep_all_features.csv')
df_original = pd.read_csv('data/blood_glucose.csv')

print("\nDataset with missing values:")
print(df_condensed.info())
print("\nDataset original:")
print(df_original.info())

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

def encode_categoricals(df):
    activities_file = 'data/activities.txt'
    with open(activities_file, 'r') as f:
        activities = f.read().splitlines()

    # Create a mapping: activity_name -> integer_code
    # Assign 0 to 'No Activity' and start coding from 1
    activity_mapping = {activity: idx + 1 for idx, activity in enumerate(activities)}
    activity_mapping['No Activity'] = 0  # Add 'No Activity' as 0

    activity_columns = [col for col in df.columns if col.startswith('activity-')]

    if not activity_columns:
        print("No activity columns found in the DataFrame.")
        return df  # Return the original DataFrame if no activity columns are found

    # Replace string labels with integer codes in each activity column
    for col in activity_columns:
        # Check if the column is of object type
        if df[col].dtype == 'object':
            df[col] = df[col].map(activity_mapping)

            # Handle unmapped activities by assigning a default value, e.g., -1
            # You can choose to handle this differently based on your needs
            df[col] = df[col].fillna(-1).astype(int)

            # Debug: Print unique values after mapping
            # print(f"Unique values in {col} after mapping:", df[col].unique())
        else:
            print(f"Column {col} is not of type 'object'. Skipping encoding for this column.")

    return df

df_original = encode_categoricals(df_original)

results_missing = train_evaluate_ridge(df_condensed, target_column, alpha=1.0)
results_original = train_evaluate_ridge(df_original, target_column, alpha=1.0)

print("Performance on Dataset with Condensed Features:")
print(f"Mean Squared Error (MSE): {results_missing['MSE']:.4f}")
print(f"R-squared (R²): {results_missing['R2']:.4f}\n")

print("Performance on Original Dataset Using Imputing:")
print(f"Mean Squared Error (MSE): {results_original['MSE']:.4f}")
print(f"R-squared (R²): {results_original['R2']:.4f}")