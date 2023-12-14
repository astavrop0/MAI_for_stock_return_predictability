import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random

DATA_PATH = os.environ.get("RESEARCH_DATA_PATH")

# Define the start and end dates for the data range
start_date = '1985-01-31'
end_date = '2018-12-31'

# Specify the proportion of test data (e.g., 10%)
test_data_size = 0.2

# Set the number of splits for K-Fold Cross Validation
KFold_split = 5

# Specify a range of alpha values for Ridge regression
alphas = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5]

# UPLOAD MONTHLY DATA
X_mef_monthly = pd.read_csv(f'{DATA_PATH}/processed/mef_monthly_data_processed.csv')
X_mai_monthly = pd.read_csv(f'{DATA_PATH}/processed/mai_monthly_data_processed.csv')
y_mkt_monthly = pd.read_csv(f'{DATA_PATH}/processed/mkt_monthly_data_processed.csv')

X_mef_monthly['date'] = pd.to_datetime(X_mef_monthly['date'])
X_mai_monthly['date'] = pd.to_datetime(X_mai_monthly['date'])
y_mkt_monthly['date'] = pd.to_datetime(y_mkt_monthly['date'])

X_mef_monthly = X_mef_monthly[(X_mef_monthly['date'] >= start_date) & (X_mef_monthly['date'] <= end_date)]
X_mai_monthly = X_mai_monthly[(X_mai_monthly['date'] >= start_date) & (X_mai_monthly['date'] <= end_date)]
y_mkt_monthly = y_mkt_monthly[(y_mkt_monthly['date'] >= start_date) & (y_mkt_monthly['date'] <= end_date)]

# Drop the 'date' column from each dataset
X_mef_monthly = X_mef_monthly.drop('date', axis=1)
X_mai_monthly = X_mai_monthly.drop('date', axis=1)
y_mkt_monthly = y_mkt_monthly.drop('date', axis=1)

# UPLOAD DAILY DATA
X_mef_daily = pd.read_csv(f'{DATA_PATH}/processed/mef_daily_data_processed.csv')
X_mai_daily = pd.read_csv(f'{DATA_PATH}/processed/mai_daily_data_processed.csv')
y_mkt_daily = pd.read_csv(f'{DATA_PATH}/processed/mkt_daily_data_processed.csv')

X_mef_daily['date'] = pd.to_datetime(X_mef_daily['date'])
X_mai_daily['date'] = pd.to_datetime(X_mai_daily['date'])
y_mkt_daily['date'] = pd.to_datetime(y_mkt_daily['date'])

X_mef_daily = X_mef_daily[(X_mef_daily['date'] >= start_date) & (X_mef_daily['date'] <= end_date)]
X_mai_daily = X_mai_daily[(X_mai_daily['date'] >= start_date) & (X_mai_daily['date'] <= end_date)]
y_mkt_daily = y_mkt_daily[(y_mkt_daily['date'] >= start_date) & (y_mkt_daily['date'] <= end_date)]

# Drop the 'date' column from each dataset
X_mef_daily = X_mef_daily.drop('date', axis=1)
X_mai_daily = X_mai_daily.drop('date', axis=1)
y_mkt_daily = y_mkt_daily.drop('date', axis=1)


X_mef_monthly = X_mef_monthly.values
X_mai_monthly = X_mai_monthly.values
y_mkt_monthly = y_mkt_monthly.values

X_mef_daily = X_mef_daily.values
X_mai_daily = X_mai_daily.values
y_mkt_daily = y_mkt_daily.values

def split_data(X, y, train_size, indices):
    # Split indices into train and test indices
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    # Split data based on the indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, y_train, X_test, y_test

def find_optimal_ridge_hyperparameters(X_train, y_train, alphas, KFold_split):
    optimal_degree = None
    optimal_alpha = None
    minimal_mse = float('inf')

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    for alpha in alphas:
        kf = KFold(n_splits=KFold_split)
        mse_arr = []

        for train_index, test_index in kf.split(X_train_std):
            X_train_n, y_train_n = X_train_std[train_index], y_train[train_index]
            X_train_v, y_train_v = X_train_std[test_index], y_train[test_index]

            # Train Ridge regression
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_n, y_train_n)

            # Predict and calculate MSE on the validation set
            y_pred = ridge.predict(X_train_v)
            mse = mean_squared_error(y_train_v, y_pred)
            mse_arr.append(mse)

        # Calculate the average MSE across KFold splits
        avg_mse = np.mean(mse_arr)

        # Update optimal hyperparameters if the current setup is better
        if avg_mse < minimal_mse:
            optimal_alpha = alpha
            minimal_mse = avg_mse

    return optimal_alpha, minimal_mse


def train_and_evaluate_ridge(X_train, y_train, X_test, y_test, alpha):
    # Train Ridge regression on the entire training set
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    # Predict on the training and test sets
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)

    # Calculate MSE for the training and test sets
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Return the MSE values and the trained model
    return mse_train, mse_test, ridge


def predict_with_ridge_batch(model, scaler, X):
    # Standardize X using the same scaler used for training
    X_std = scaler.transform(X)

    # Predict the target values using the trained Ridge model
    predicted_y = model.predict(X_std)

    return predicted_y


# Training and evaluating Ridge model

# Set the random seed for reproducibility
np.random.seed(11)
random.seed(11)

# Generate a list of 10 different random permutations
all_indices = [np.random.permutation(X_mef_monthly.shape[0]) for _ in range(10)]

#lists to store the error of each iteration
MEF_errors_train = []
MEF_errors_test = []
MAI_errors_train = []
MAI_errors_test = []
 
train_size = int((1-test_data_size) * X_mef_monthly.shape[0])

for i in range(10):
    # Split data into training and test sets
    indices = all_indices[i]
    X_train_mef, y_train_mef, X_test_mef, y_test_mef = split_data(X_mef_monthly, y_mkt_monthly, train_size, indices)
    X_train_mai, y_train_mai, X_test_mai, y_test_mai = split_data(X_mai_monthly, y_mkt_monthly, train_size, indices)

    # Standardize both training and test data using a scaler
    scaler = StandardScaler()
    X_train_std_mef = scaler.fit_transform(X_train_mef)
    X_test_std_mef = scaler.transform(X_test_mef)  
    X_train_std_mai = scaler.fit_transform(X_train_mai)
    X_test_std_mai = scaler.transform(X_test_mai)

    # Find optimal alpha using cross-validation
    optimal_alpha_mef, minimal_mse_mef = find_optimal_ridge_hyperparameters(X_train_std_mef, y_train_mef, alphas, KFold_split)    
    optimal_alpha_mai, minimal_mse_mai = find_optimal_ridge_hyperparameters(X_train_std_mai, y_train_mai, alphas, KFold_split)

    # Evaluate the model on the test set (MEF)
    mse_train_mef, mse_test_mef, ridge_model_mef = train_and_evaluate_ridge(X_train_std_mef, y_train_mef, X_test_std_mef, y_test_mef, optimal_alpha_mef)
    y_pred_mef = ridge_model_mef.predict(X_test_std_mef)
    rmse_train_mef = mse_train_mef**(1/2)
    rmse_test_mef = mse_test_mef**(1/2)
 
    #add the resulting error in the list
    MEF_errors_train.append(rmse_train_mef)
    MEF_errors_test.append(rmse_test_mef)
    
    # Evaluate the model on the test set (MAI)
    mse_train_mai, mse_test_mai, ridge_model_mai = train_and_evaluate_ridge(X_train_std_mai, y_train_mai, X_test_std_mai, y_test_mai, optimal_alpha_mai)
    y_pred_mai = ridge_model_mai.predict(X_test_std_mai)
    rmse_train_mai = mse_train_mai**(1/2)
    rmse_test_mai = mse_test_mai**(1/2)
    
    MAI_errors_train.append(rmse_train_mai)
    MAI_errors_test.append(rmse_test_mai)

MEF_avg_rmse_train = sum(MEF_errors_train) / len(MEF_errors_train)   
MEF_avg_rmse_test = sum(MEF_errors_test) / len(MEF_errors_test)
print('MEF Linear Predictor')
print(f"Average Train RMSE for MEF monthly data: {MEF_avg_rmse_train}")
print(f"Average Test RMSE for MEF monthly data: {MEF_avg_rmse_test}")
print("")

MAI_avg_rmse_train = sum(MAI_errors_train) / len(MAI_errors_train)
MAI_avg_rmse_test = sum(MAI_errors_test) / len(MAI_errors_test)
print('MAI Linear Predictor')
print(f"Average Train RMSE for MAI monthly data: {MAI_avg_rmse_train}")
print(f"Average Test RMSE for MAI monthly data: {MAI_avg_rmse_test}")
print("")


#Plot Average train and test errors

labels = ['Train', 'Test']
MEF_rmse_values = [MEF_avg_rmse_train, MEF_avg_rmse_test]
MAI_rmse_values = [MAI_avg_rmse_train, MAI_avg_rmse_test]

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.bar(labels, MEF_rmse_values, color=['#1f77b4', '#ff7f0e'])
plt.xlabel('Dataset')
plt.ylabel('Average RMSE')
plt.title('Average RMSE for Train and Test Sets (MEF data)')

plt.subplot(1,2,2)
plt.bar(labels, MAI_rmse_values, color=['#1f77b4', '#ff7f0e'])
plt.xlabel('Dataset')
plt.ylabel('Average RMSE')
plt.title('Average RMSE for Train and Test Sets (MAI data)')

plt.show()


# Boxplots for distribution of errors over the 10 iterations
plt.figure(figsize=(14,11))
plt.subplot(2,2,1)
sns.boxplot(data=[MEF_errors_train, MEF_errors_test])
plt.xticks([0, 1], ['Train', 'Test'])
plt.ylabel('RMSE')
plt.title('Distribution of RMSE over Iterations (MEF data)')

plt.subplot(2,2,2)
sns.boxplot(data=[MAI_errors_train, MAI_errors_test])
plt.xticks([0, 1], ['Train', 'Test'])
plt.ylabel('RMSE')
plt.title('Distribution of RMSE over Iterations (MAI data)')

plt.subplot(2, 2, 3)
plt.plot(MEF_errors_train, label='Train RMSE')
plt.plot(MEF_errors_test, label='Test RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('RMSE over Iterations (MEF data)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(MAI_errors_train, label='Train RMSE')
plt.plot(MAI_errors_test, label='Test RMSE')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.title('RMSE over Iterations (MAI data)')
plt.legend()
plt.show()


# For MEF data
plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.scatter(y_test_mef, y_pred_mef)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (MEF data)')
plt.plot([y_test_mef.min(), y_test_mef.max()], [y_test_mef.min(), y_test_mef.max()], 'k--', lw=3)


# For MAI data
plt.subplot(1,2,2)
plt.scatter(y_test_mai, y_pred_mai)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (MAI data)')
plt.plot([y_test_mai.min(), y_test_mai.max()], [y_test_mai.min(), y_test_mai.max()], 'k--', lw=3)
plt.show()


# Set the random seed for reproducibility
np.random.seed(8)
random.seed(8)

train_size = int((1-test_data_size) * X_mai_daily.shape[0])
indices = np.random.permutation(X_mai_daily.shape[0])

np.random.shuffle(indices)
X_train_mai_d, y_train_mai_d, X_test_mai_d, y_test_mai_d = split_data(X_mai_daily, y_mkt_daily, train_size, indices)
X_train_mef_d, y_train_mef_d, X_test_mef_d, y_test_mef_d = split_data(X_mef_daily, y_mkt_daily, train_size, indices)

scaler = StandardScaler()
X_train_std_mai_d = scaler.fit_transform(X_train_mai_d)
X_test_std_mai_d = scaler.transform(X_test_mai_d) 

X_train_std_mef_d = scaler.fit_transform(X_train_mef_d)
X_test_std_mef_d = scaler.transform(X_test_mef_d) 

optimal_alpha_mai, minimal_mse_mai = find_optimal_ridge_hyperparameters(X_train_std_mai_d, y_train_mai_d, alphas, KFold_split)    
optimal_alpha_mef, minimal_mse_mef = find_optimal_ridge_hyperparameters(X_train_std_mef_d, y_train_mef_d, alphas, KFold_split)    
  
mse_train_mai, mse_test_mai, ridge_model_mai = train_and_evaluate_ridge(X_train_std_mai_d, y_train_mai_d, X_test_std_mai_d, y_test_mai_d, optimal_alpha_mai)
rmse_train_mai = mse_train_mai**(1/2)
rmse_test_mai = mse_test_mai**(1/2)

print(f"Train RMSE for MAI daily data: {rmse_train_mai}")
print(f"Test RMSE for MAI daily data: {rmse_test_mai}")
print("")

mse_train_mef, mse_test_mef, ridge_model_mef = train_and_evaluate_ridge(X_train_std_mef_d, y_train_mef_d, X_test_std_mef_d, y_test_mef_d, optimal_alpha_mef)
rmse_train_mef = mse_train_mef**(1/2)
rmse_test_mef = mse_test_mef**(1/2)

print(f"Train RMSE for MEF daily data: {rmse_train_mef}")
print(f"Test RMSE for MEF daily data: {rmse_test_mef}")
print("")

# Train and Test RMSE values
rmse_values_mai = [rmse_train_mai, rmse_test_mai]
rmse_values_mef = [rmse_train_mef, rmse_test_mef]

# Labels for the bar plots
labels = ['Train', 'Test']

# Create two subplots in a row
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Plot for MAI data
axs[0].bar(labels, rmse_values_mai, color=['#1f77b4', '#ff7f0e'])
axs[0].set_title('MAI Daily Data RMSE')
axs[0].set_ylabel('RMSE')
axs[0].set_ylim([0, max(rmse_values_mai + rmse_values_mef) * 1.1])  # Adjust y-axis limits

# Plot for MEF data
axs[1].bar(labels, rmse_values_mef, color=['#1f77b4', '#ff7f0e'])
axs[1].set_title('MEF Daily Data RMSE')
axs[1].set_ylim([0, max(rmse_values_mai + rmse_values_mef) * 1.1])  # Adjust y-axis limits

plt.show()


y_pred_mai_d = ridge_model_mai.predict(X_test_std_mai_d)
y_pred_mef_d = ridge_model_mef.predict(X_test_std_mef_d)

# Create two subplots in a row for scatter plots
fig, axs = plt.subplots(1, 2, figsize=(18, 6))

# Scatter plot for MAI data
axs[0].scatter(y_test_mai_d, y_pred_mai_d)
axs[0].plot([y_test_mai_d.min(), y_test_mai_d.max()], [y_test_mai_d.min(), y_test_mai_d.max()], 'k--', lw=3)
axs[0].set_xlabel('Actual Values')
axs[0].set_ylabel('Predicted Values')
axs[0].set_title('Predicted vs Actual Values (MAI Daily Data)')

# Scatter plot for MEF data
axs[1].scatter(y_test_mef_d, y_pred_mef_d)
axs[1].plot([y_test_mef_d.min(), y_test_mef_d.max()], [y_test_mef_d.min(), y_test_mef_d.max()], 'k--', lw=3)
axs[1].set_xlabel('Actual Values')
axs[1].set_ylabel('Predicted Values')
axs[1].set_title('Predicted vs Actual Values (MEF Daily Data)')

plt.show()




