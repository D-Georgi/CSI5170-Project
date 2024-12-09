import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer

"""
Section 1: Data preprocessing (loading and scaling)
"""
# Grab test dataset
df_original = pd.read_csv('data/blood_glucose.csv')

# Identify target column
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

def preprocess(df, strategy):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer_arr = imputer.fit_transform(df)

    imputed_df = pd.DataFrame(imputer_arr, columns=df.columns)

    X = imputed_df.drop(columns=[target_column])
    y = imputed_df[target_column]

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    return X, y

# Encode origiginal dataset
df_original = encode_categoricals(df_original)
# Preprocess encoded dataset using mean strategy
X, y = preprocess(df_original, 'mean')

"""
Section 2: Regularized Linear Regression
"""

# Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)

# Declare variables
ridge_time = 0
ridge_mse = 0
ridge_r2 = 0

def ridgeOptimization(): 
    # Regression model
    model = Ridge()

    # Create hyperparameter coefficients
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'max_iter': [1000, 2000, 3000],  # Max iterations for the solver
    }

    scoring = {'MSE': 'neg_mean_squared_error', 'R2': 'r2'}

    # Run GridSearchCV to find best coefficient
    ridgeGrid = GridSearchCV(model, param_grid, cv=5, scoring=scoring, refit='MSE')

    # Fit the training data to the grid search
    ridgeGrid.fit(X_train_reg, y_train_reg)

    # Grab the best estimator
    bestEstimator = ridgeGrid.best_estimator_
    print("Best Estimator: " + str(bestEstimator))

    # Grab the grid search results into a dataframe
    results = pd.DataFrame(ridgeGrid.cv_results_)
    results['alpha'] = results['param_alpha'].astype(str)
    results['max_iter'] = results['param_max_iter'].astype(str)
    results['MSE'] = -results['mean_test_MSE']
    results['R2'] = results['mean_test_R2']

    # Grab only the relevant columns
    results = results[['mean_fit_time', 'alpha', 'max_iter', 'MSE', 'R2']]

    # Cast the alpha and max_iter values to numerics
    results['alpha'] = pd.to_numeric(results['alpha'])
    results['max_iter'] = pd.to_numeric(results['max_iter'])

    # Grab only the line best estimator row of the results
    bestModel = results[(results['alpha'] == bestEstimator.alpha) & (results['max_iter'] == bestEstimator.max_iter)]

    # Print the training time and MSE/R2 for the best estimator
    ridge_time = bestModel['mean_fit_time'].values[0]
    ridge_mse = bestModel['MSE'].values[0]
    ridge_r2 = bestModel['R2'].values[0]
    print(f'Training time for ridge regression: {ridge_time:0.4f} seconds')
    print(f'MSE: {ridge_mse:0.4f}, R2: {ridge_r2:0.4f}')

    return 1

# Find optimal hpyerparameters, MSE, and R2
ridgeOptimization()

"""
Section 3: Convolutional Neural Network for Regression
"""

# Reshape the data into 3d to feed into Conv1D
X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing data
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Declare variables
best_loss = 0
best_epoch = 0
best_batch_size = 0
cnn_time = 0
cnn_best_mse = 0
cnn_best_r2 = 0
best_mse_vals = 0
best_val_mse_vals = 0
best_r2_vals = 0
best_val_r2_vals = 0

def cnnModel():
    # CNN Model Creation/definition. Using relu activation
    model = models.Sequential([
    layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(128, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Conv1D(256, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_squared_error', 'r2_score'])

    return model

# Create hyperparameter values
epoch = [10, 25, 50]
batchSize = [32, 64, 128]

# Initialize best loss metric
best_loss = float('inf')

# EarlyStopping callback to monitor validation MSE (patience of 3)
earlyStop = EarlyStopping(monitor='val_mean_squared_error', patience=3, restore_best_weights=True)

# Loop through combinations of epochs and batch sizes
for epochs in epoch:
    for batch_size in batchSize:
        print("-----" * 35)
        print(f"Training with epochs={epochs} and batch_size={batch_size}")

        # Create the model for each combination
        model = cnnModel()

        # Time the training of the model
        cnn_start_time = time.perf_counter()
        history = model.fit(
            X_train_cnn, y_train_cnn,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_cnn, y_test_cnn),
            callbacks=[earlyStop],
            verbose=0
        )
        cnn_elapsed_time = time.perf_counter() - cnn_start_time

        # Evaluate the model
        cnn_loss, cnn_mse, cnn_r2 = model.evaluate(X_test_cnn, y_test_cnn)

        print(f'Training time for CNN: {cnn_elapsed_time:0.4f} seconds')
        print(f"Epochs: {epochs}, Batch Size: {batch_size} => Loss: {cnn_loss:.4f}, MSE: {cnn_mse:.4f}, R2: {cnn_r2:.4f}")
        
        # Update best hyperparameters if current model performs better based on loss
        if cnn_loss < best_loss:
            best_loss = cnn_loss
            best_epoch = epochs
            best_batch_size = batch_size
            cnn_time = cnn_elapsed_time
            cnn_best_mse = cnn_mse
            cnn_best_r2 = cnn_r2
            best_mse_vals = history.history['mean_squared_error']
            best_val_mse_vals = history.history['val_mean_squared_error']
            best_r2_vals = history.history['r2_score']
            best_val_r2_vals = history.history['val_r2_score']

# Print out the best hyperparameters and their MSE and R2 values
print(f"\nBest Epochs: {best_epoch}, Best Batch Size: {best_batch_size}")
print(f"Best Validation MSE: {cnn_best_mse:0.4f}")
print(f"Best Validation R2: {cnn_best_r2:0.4f}")

# Plotting the MSE values for train and validation
epochs = range(1, len(best_mse_vals) + 1)
plt.plot(epochs, best_mse_vals, 'r', label='Training MSE')
plt.plot(epochs, best_val_mse_vals, 'b', label='Validation MSE')
plt.title('Training and validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plotting the R2 values for train and validation
epochs = range(1, len(best_r2_vals) + 1)
plt.plot(epochs, best_r2_vals, 'r', label='Training R2')
plt.plot(epochs, best_val_r2_vals, 'b', label='Validation R2')
plt.title('Training and validation R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.show()

"""
Section 4: Analyze and Compare
"""

print('\nTraining Time Comparrison:')
print(f'Training time for ridge regression: {ridge_time:0.4f} seconds')
print(f'Training time for CNN: {cnn_time:0.4f} seconds')

# Create bar graph of the training time of each model
data = {'Model': ['Ridge Regression', 'CNN Regression'], 'Training Time': [ridge_time, cnn_time]}
df = pd.DataFrame(data)
plt.bar(df['Model'], df['Training Time'])
plt.xlabel('Model')
plt.ylabel('Training Time (s)')
plt.title('Training Time For Each Model')
plt.show()

print('\nMSE Comparrison:')
print(f'Ridge MSE: {ridge_mse:0.4f}')
print(f'CNN Validation MSE: {cnn_best_mse:0.4f}')

# Create bar graph of the MSE of each model
data = {'Model': ['Ridge Regression', 'CNN Regression'], 'MSE': [ridge_mse, cnn_best_mse]}
df = pd.DataFrame(data)
plt.bar(df['Model'], df['MSE'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE For Each Model')
plt.show()

print('\nR2 Comparrison:')
print(f'Ridge R2: {ridge_r2:0.4f}')
print(f'CNN Validation R2: {cnn_best_r2:0.4f}')

# Create bar graph of the R2 of each model
data = {'Model': ['Ridge Regression', 'CNN Regression'], 'R2': [ridge_r2, cnn_best_r2]}
df = pd.DataFrame(data)
plt.bar(df['Model'], df['R2'])
plt.xlabel('Model')
plt.ylabel('R2')
plt.title('R2 For Each Model')
plt.show()