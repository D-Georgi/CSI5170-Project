import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

df_original = pd.read_csv('data/blood_glucose.csv')
print("\nOriginal dataset:")
print(df_original.info())

target_column = 'bg+1:00'

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

def preprocess_and_run(df):
    df_fill_zeros = df.fillna(0)

    df_fill_zeros.info()

    X = df_fill_zeros.drop(columns=[target_column])
    y = df_fill_zeros[target_column]

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

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ridge = Ridge(alpha=1.0)

    ridge.fit(X, y)
    y_pred_fill = ridge.predict(X_test)

    mse_fill = mean_squared_error(y_test, y_pred_fill)
    r2_fill = r2_score(y_test, y_pred_fill)

    print("Performance on Dataset with Filled Missing Values:")
    print(f"Mean Squared Error (MSE): {mse_fill:.4f}")
    print(f"R-squared (R²): {r2_fill:.4f}")

df_original = encode_categoricals(df_original)
results_original = preprocess_and_run(df_original)