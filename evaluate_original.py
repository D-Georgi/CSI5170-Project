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

    df = df.drop(columns=['id'])
    df = df.drop(columns=['p_num'])
    df = df.drop(columns=['time'])
    return df

def preprocess_and_run(df, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer_arr = imputer.fit_transform(df)

    imputed_df = pd.DataFrame(imputer_arr, columns=df.columns)

    X = imputed_df.drop(columns=[target_column])
    y = imputed_df[target_column]

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

    print(f"Performance on Dataset with {strategy} Imputed Missing Values:")
    print(f"Mean Squared Error (MSE): {mse_fill:.4f}")
    print(f"R-squared (R²): {r2_fill:.4f}")

df_original = encode_categoricals(df_original)
preprocess_and_run(df_original, 'mean')
preprocess_and_run(df_original, 'median')
preprocess_and_run(df_original, 'most_frequent')