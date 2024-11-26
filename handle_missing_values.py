import pandas as pd
import numpy as np
import re

# --------------------------------------------
# Step 1: Read and Map Activities
# --------------------------------------------

# Read the activities.txt file to get activity names
activities_file = 'data/activities.txt'
with open(activities_file, 'r') as f:
    activities = f.read().splitlines()

# Create a mapping: activity_name -> integer_code
# Assign 0 to 'No Activity' and start coding from 1
activity_mapping = {activity: idx + 1 for idx, activity in enumerate(activities)}
activity_mapping['No Activity'] = 0  # Add 'No Activity' as 0

print("Activity Mapping:")
for activity, code in activity_mapping.items():
    print(f" - {activity}: {code}")

# --------------------------------------------
# Step 2: Load the Dataset
# --------------------------------------------

# Load the dataset with low_memory=False to handle mixed types
data_path = 'data/blood_glucose.csv'
df = pd.read_csv(data_path, low_memory=False)
print("\nInitial Data Shape:", df.shape)
print("Columns:", df.columns.tolist())

# --------------------------------------------
# Step 3: Parse Feature Names
# --------------------------------------------

# Function to parse feature names with logging
def parse_feature(name):
    pattern = r'^(?P<feature_type>\w+)-(?P<offset>\d+):(?P<minute>\d+)$'
    match = re.match(pattern, name)
    if match:
        return match.groupdict()
    else:
        return None

# Extract feature information
# Assuming the first three columns are 'id', 'p_num', 'time' and last column is 'bg+1:00'
feature_info = df.columns[3:-1].map(parse_feature)  # Exclude 'bg+1:00'
feature_df = pd.DataFrame(feature_info.tolist(), index=df.columns[3:-1])

# Identify columns that failed to parse
failed_parsing = feature_df[feature_df['feature_type'].isnull()].index.tolist()
if failed_parsing:
    print("\nColumns that failed to parse and will be excluded:")
    for col in failed_parsing:
        print(f" - {col}")
else:
    print("\nAll columns parsed successfully.")

# Exclude columns that failed to parse
valid_features = feature_df.dropna(subset=['feature_type', 'offset', 'minute']).index.tolist()
feature_df = feature_df.dropna(subset=['feature_type', 'offset', 'minute'])
print(f"\nNumber of valid features: {len(valid_features)}")

# --------------------------------------------
# Step 4: Compute Total Minutes and Assign Bins
# --------------------------------------------

# Function to convert offset and minute to total minutes
def compute_total_minutes(row):
    try:
        hours = int(row['offset'])
        minutes = int(row['minute'])
        return hours * 60 + minutes
    except Exception as e:
        print(f"Error computing total_minutes for row: {e}")
        return np.nan

# Compute total_minutes using both 'offset' and 'minute'
feature_df['total_minutes'] = feature_df.apply(compute_total_minutes, axis=1)

# Check for any NaN in 'total_minutes' after conversion
nan_minutes = feature_df['total_minutes'].isnull().sum()
if nan_minutes > 0:
    print(f"\nWarning: {nan_minutes} features have invalid 'total_minutes' and will be excluded.")
    # Exclude these features
    feature_df = feature_df.dropna(subset=['total_minutes'])
    valid_features = feature_df.index.tolist()

# Define 30-minute bins
bin_size = 30
feature_df['bin'] = (feature_df['total_minutes'] // bin_size) * bin_size
print("\nFeature DataFrame Head:")
print(feature_df.head())

# --------------------------------------------
# Step 5: Identify Numerical and Categorical Features
# --------------------------------------------

numerical_features = ['bg', 'insulin', 'carbs', 'hr', 'steps', 'cals']
categorical_features = ['activity']

# --------------------------------------------
# Step 6: Aggregate Numerical Features
# --------------------------------------------

# Initialize a dictionary to hold aggregated data
aggregated_data = {}

for feature in numerical_features:
    feature_cols = feature_df[feature_df['feature_type'] == feature].index.tolist()
    if not feature_cols:
        print(f"\nNo columns found for numerical feature '{feature}'. Skipping.")
        continue
    # Extract relevant columns
    sub_df = df[feature_cols]
    # Convert to numeric (if not already)
    sub_df = sub_df.apply(pd.to_numeric, errors='coerce')
    # Get bins for each column
    bins = feature_df[feature_df['feature_type'] == feature]['bin'].tolist()
    # Assign bin to each column by renaming
    sub_df.columns = bins
    # Group by bin and compute mean
    agg_numerical = sub_df.groupby(sub_df.columns, axis=1).mean()
    # Rename columns to include feature type and bin
    agg_numerical.columns = [f"{feature}_{int(col)}min" for col in agg_numerical.columns]
    # Add to aggregated data
    aggregated_data.update(agg_numerical.to_dict(orient='list'))
    print(f"\nAggregated numerical feature '{feature}' with bins: {agg_numerical.columns.tolist()}")

# Convert aggregated numerical data to DataFrame
agg_numerical_df = pd.DataFrame(aggregated_data)
print("\nAggregated Numerical Features Shape:", agg_numerical_df.shape)
print("Aggregated Numerical Features Head:")
print(agg_numerical_df.head())

# --------------------------------------------
# Step 7: Aggregate Categorical Features (Activity)
# --------------------------------------------

# Reset aggregated_data for categorical features
aggregated_data = {}

for feature in categorical_features:
    feature_cols = feature_df[feature_df['feature_type'] == feature].index.tolist()
    if not feature_cols:
        print(f"\nNo columns found for categorical feature '{feature}'. Skipping.")
        continue
    # Extract relevant columns
    sub_df = df[feature_cols]
    # Get bins for each column
    bins = feature_df[feature_df['feature_type'] == feature]['bin'].tolist()
    # Assign bin to each column by renaming
    sub_df.columns = bins
    # Map activity names to integer codes
    sub_df = sub_df.replace(activity_mapping)
    # Any missing values (no activity) are set to 0
    sub_df = sub_df.fillna(0).astype(int)
    # Since multiple columns might map to the same bin, group by bin and decide how to handle multiple activities
    # Here, we'll take the maximum code in each bin as a representative activity
    # Alternatively, you could define a different aggregation method
    agg_activity = sub_df.groupby(sub_df.columns, axis=1).max()
    # Rename columns to include feature type and bin
    agg_activity.columns = [f"{feature}_{int(col)}min" for col in agg_activity.columns]
    # Add to aggregated data
    aggregated_data.update(agg_activity.to_dict(orient='list'))
    print(f"\nAggregated categorical feature '{feature}' with bins: {agg_activity.columns.tolist()}")

# Convert aggregated categorical data to DataFrame
agg_categorical_df = pd.DataFrame(aggregated_data)
print("\nAggregated Categorical Features Shape:", agg_categorical_df.shape)
print("Aggregated Categorical Features Head:")
print(agg_categorical_df.head())

# --------------------------------------------
# Step 8: Combine Aggregated Features
# --------------------------------------------

# Combine aggregated numerical and categorical features
agg_df = pd.concat([agg_numerical_df, agg_categorical_df], axis=1)
print("\nCombined Aggregated Features Shape:", agg_df.shape)
print("Combined Aggregated Features Head:")
print(agg_df.head())

# --------------------------------------------
# Step 9: Handle Missing Values
# --------------------------------------------

# Calculate degree of missingness
missing_percent = agg_df.isnull().mean() * 100
print("\nMissingness per Feature:")
print(missing_percent.sort_values(ascending=False))

# Define a threshold for missingness
threshold = 70.0

# Features to drop
features_to_drop = missing_percent[missing_percent > threshold].index.tolist()
print("\nFeatures to Drop Due to High Missingness:", features_to_drop)

# Drop these features
if features_to_drop:
    agg_df = agg_df.drop(columns=features_to_drop)
    print(f"\nAggregated Data Shape after Dropping: {agg_df.shape}")
else:
    print("\nNo features to drop based on the missingness threshold.")

# --------------------------------------------
# Step 10: Impute Remaining Missing Values
# --------------------------------------------

# Impute numerical and categorical features
numerical_cols = agg_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = agg_df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical features with mean
if numerical_cols:
    agg_df[numerical_cols] = agg_df[numerical_cols].fillna(agg_df[numerical_cols].mean())

# Impute categorical features with mode (if any)
for col in categorical_cols:
    mode = agg_df[col].mode()
    if not mode.empty:
        agg_df[col] = agg_df[col].fillna(mode[0])
    else:
        # If mode is empty, fill with a placeholder
        agg_df[col] = agg_df[col].fillna('Unknown')

# Verify no missing values remain
print("\nMissing Values After Imputation:")
print(agg_df.isnull().sum())

print("\nNo one-hot encoding applied to 'activity' feature.")

# --------------------------------------------
# Step 12: Combine with Essential Non-Aggregated Columns
# --------------------------------------------

# Combine with essential non-aggregated columns
final_df = pd.concat([df[['id', 'p_num', 'time', 'bg+1:00']], agg_df], axis=1)
print("\nFinal DataFrame Shape:", final_df.shape)
print("Final DataFrame Head:")
print(final_df.head())

# --------------------------------------------
# Step 13: Save the Processed Dataset
# --------------------------------------------

# Save to a new CSV
final_df.to_csv('data/blood_glucose_30min_avg.csv', index=False)
print("\nProcessed data saved to 'data/blood_glucose_30min_avg.csv'")

