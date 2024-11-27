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

# Load the data
df_no_missing = pd.read_csv('data/blood_glucose_avgs_all_missing_values.csv')

# Inspect dataset
df_no_missing.info()

# Identify target column
target_column = 'bg+1:00'

X = df_no_missing.drop(columns=[target_column])
y = df_no_missing[target_column]

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

# Scale feature data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

"""
Section 2: Regularized Linear Regression
"""

# Split the data into training and testing sets
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=42)

def ridgeHyperparameterOptimization(): 
    # Regression model
    model = Ridge()

    # Create hyperparameter coefficients
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'max_iter': [1000, 2000, 3000],  # Max iterations for the solver
    }

    # Run GridSearchCV to find best coefficient
    ridgeGrid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

    # Fit the training data to the grid search
    ridgeGrid.fit(X_train_reg, y_train_reg)

    # Print out the best alpha value
    print("Best Estimator: " + str(ridgeGrid.best_estimator_))
    alpha = ridgeGrid.best_estimator_.alpha
    max_iter = ridgeGrid.best_estimator_.max_iter
    tol = ridgeGrid.best_estimator_.tol

    return {'alpha': alpha, 'max_iter': max_iter}

# Find optimal alpha
optimalParameters = ridgeHyperparameterOptimization()

# Create the ridge regression model
ridge = Ridge(alpha=optimalParameters['alpha'], max_iter=optimalParameters['max_iter'])

# Time the training of the model
ridge_start_time = time.perf_counter()
ridgeModel = ridge.fit(X_train_reg, y_train_reg)
ridge_elapsed_time = time.perf_counter() - ridge_start_time

# Predict on the test data
y_pred = ridgeModel.predict(X_test_reg)

# Calculate performance metrics
ridge_mse = mean_squared_error(y_test_reg, y_pred)
ridge_r2 = r2_score(y_test_reg, y_pred)

print(f'Training time for ridge regression: {ridge_elapsed_time:0.4f} seconds')
print(f'MSE: {ridge_mse:0.4f}, R2: {ridge_r2:0.4f}')

"""
Section 3: Convolutional Neural Network for Regression
"""

# Reshape the data into 3d to feed into Conv1D
X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))

# Split the data into training and testing data
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

def cnnModel():
    # CNN Model Creation/definition. Using relu activation and softmax for the last layer
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
earlyStop = EarlyStopping(monitor='val_mean_squared_error', patience=5, restore_best_weights=True)

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
            cnn_elapsed_time = cnn_elapsed_time
            cnn_best_mse = cnn_mse
            cnn_best_r2 = cnn_r2
            best_mse_vals = history.history['mean_squared_error']
            best_val_mse_vals = history.history['val_mean_squared_error']
            best_r2_vals = history.history['r2_score']
            best_val_r2_vals = history.history['val_r2_score']

# Print out the best hyperparameters and their MSE and R2 values
print(f"\nBest Epochs: {best_epoch}, Best Batch Size: {best_batch_size}")
print(f"Best MSE: {best_mse_vals}, Best Val MSE: {best_val_mse_vals}")
print(f"Best R2: {best_r2_vals}, Best Val R2: {best_val_r2_vals}")

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
print(f'Training time for ridge regression: {ridge_elapsed_time:0.4f} seconds')
print(f'Training time for CNN: {cnn_elapsed_time:0.4f} seconds')

print('\nMSE Comparrison:')
print(f'Ridge MSE: {ridge_mse:0.4f}')
print(f'CNN MSE: {cnn_best_mse:0.4f}')
print(f'CNN MSE (list): {max(best_mse_vals):0.4f}')
print(f'CNN Val MSE: {max(best_val_mse_vals):0.4f}')

print('\nR2 Comparrison:')
print(f'R2: {ridge_r2:0.4f}')
print(f'CNN MSE: {cnn_best_r2:0.4f}')
print(f'CNN MSE (list): {max(best_r2_vals):0.4f}')
print(f'CNN Val MSE: {max(best_val_r2_vals):0.4f}')