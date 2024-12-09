import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor  # Changed to Regressor
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Load the original dataset
df_original = pd.read_csv('data/blood_glucose_30min_avg_keep_missing.csv')
print("\nOriginal dataset:")
print(df_original.info())
df_original = df_original.drop(columns='id')
df_original = df_original.drop(columns='p_num')
df_original = df_original.drop(columns='time')
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

    # Drop unnecessary columns
    df = df.drop(columns=['id', 'p_num', 'time'])
    return df

# Encode categorical activity columns
df_encoded = df_original

# Separate features and target
X = df_encoded.drop(columns=[target_column])
y = df_encoded[target_column]

# Check the target variable
print("\nTarget variable statistics:")
print(y.describe())

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature selection based on variance
selector = VarianceThreshold(threshold=0.01)
X_var = selector.fit_transform(X_imputed)

# Get the selected feature names
selected_features = X_imputed.columns[selector.get_support()]

# Convert the NumPy array back to DataFrame with selected feature names
X_var_df = pd.DataFrame(X_var, columns=selected_features)

# Feature selection based on importance using RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_var_df, y)

# Get feature importances
importances = model.feature_importances_
feature_importances = pd.Series(importances, index=X_var_df.columns).sort_values(ascending=False)

# Select top N features (e.g., top 50)
top_n = 50
important_features = feature_importances.head(top_n).index
X_selected = X_var_df[important_features]

print(f"Selected top {top_n} features based on importance.")

print(important_features)

# Dimensionality reduction using PCA
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_selected)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['target'] = y.values

# Visualization
plt.figure(figsize=(10, 8))
scatter = sns.scatterplot(x='PC1', y='PC2', hue='target', palette='viridis', data=pca_df, alpha=0.7)
plt.title('PCA Projection of Blood Glucose Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Blood Glucose Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
