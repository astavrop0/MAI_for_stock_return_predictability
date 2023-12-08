import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

X_mef_monthly = X_mef_monthly.values
X_mai_monthly = X_mai_monthly.values
y_mkt_monthly = y_mkt_monthly.values

# UPLOAD DAILY DATA
X_mai_daily = pd.read_csv(f'{DATA_PATH}/processed/mai_daily_data_processed.csv')
y_mkt_daily = pd.read_csv(f'{DATA_PATH}/processed/mkt_daily_data_processed.csv')

X_mai_daily['date'] = pd.to_datetime(X_mai_daily['date'])
y_mkt_daily['date'] = pd.to_datetime(y_mkt_daily['date'])

X_mai_daily = X_mai_daily[(X_mai_daily['date'] >= start_date) & (X_mai_daily['date'] <= end_date)]
y_mkt_daily = y_mkt_daily[(y_mkt_daily['date'] >= start_date) & (y_mkt_daily['date'] <= end_date)]

# Drop the 'date' column from each dataset
X_mai_daily = X_mai_daily.drop('date', axis=1)
y_mkt_daily = y_mkt_daily.drop('date', axis=1)

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
    #print(f" MEF Coefficients: {ridge.coef_}")


    # Predict on the training and test sets
    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)

    # Calculate and print MSE for the training and test sets
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    return mse_train, mse_test

def predict_with_ridge(model, scaler, new_X):
    # Standardize the new_X using the same scaler used for training
    new_X_std = scaler.transform(new_X.reshape(1, -1))

    # Predict the target value using the trained Ridge model
    predicted_y = model.predict(new_X_std)

    return predicted_y[0]

# Training and evaluating Ridge model: MEF

MEF_errors_train = []
MEF_errors_test = []
MAI_errors_train = []
MAI_errors_test = []
 
train_size = int((1-test_data_size) * X_mef_monthly.shape[0])
indices = np.random.permutation(X_mef_monthly.shape[0])

for i in range(10):
    # Split data into training and test sets
    np.random.shuffle(indices)
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

    minimal_rmse_mef = minimal_mse_mef**(1/2)
    #print(f"Optimal alpha for MEF data: {optimal_alpha_mef}, Minimal MSE: {minimal_rmse_mef}")    
    minimal_rmse_mai = minimal_mse_mai**(1/2)
    #print(f"Optimal alpha for MAI data: {optimal_alpha_mai}, Minimal MSE: {minimal_rmse_mai}")
    
    # Evaluate the model on the test set
    mse_train_mef, mse_test_mef = train_and_evaluate_ridge(X_train_std_mef, y_train_mef, X_test_std_mef, y_test_mef, optimal_alpha_mef)
    rmse_train_mef = mse_train_mef**(1/2)
    rmse_test_mef = mse_test_mef**(1/2)
    
    MEF_errors_train.append(rmse_train_mef)
    MEF_errors_test.append(rmse_test_mef)

    mse_train_mai, mse_test_mai = train_and_evaluate_ridge(X_train_std_mai, y_train_mai, X_test_std_mai, y_test_mai, optimal_alpha_mai)
    rmse_train_mai = mse_train_mai**(1/2)
    rmse_test_mai = mse_test_mai**(1/2)
    
    MAI_errors_train.append(rmse_train_mai)
    MAI_errors_test.append(rmse_test_mai)

    # Save the trained model and scaler for later use
    ridge_scaler = scaler

    
MEF_avg_rmse_train = sum(MEF_errors_train) / len(MEF_errors_train)   
MEF_avg_rmse_test = sum(MEF_errors_test) / len(MEF_errors_test)
print('MEF Linear Predictor')
print(f"Average Train RMSE for MEF data: {MEF_avg_rmse_train}")
print(f"Average Test RMSE for MEF data: {MEF_avg_rmse_test}")
print("")

MAI_avg_rmse_train = sum(MAI_errors_train) / len(MAI_errors_train)
MAI_avg_rmse_test = sum(MAI_errors_test) / len(MAI_errors_test)
print('MAI Linear Predictor')
print(f"Average Train RMSE for MAI data: {MAI_avg_rmse_train}")
print(f"Average Test RMSE for MAI data: {MAI_avg_rmse_test}")
print("")

train_size = int((1-test_data_size) * X_mai_daily.shape[0])
indices = np.random.permutation(X_mai_daily.shape[0])

np.random.shuffle(indices)
X_train, y_train, X_test, y_test = split_data(X_mai_daily, y_mkt_daily, train_size, indices)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)  

optimal_alpha, minimal_mse = find_optimal_ridge_hyperparameters(X_train_std, y_train, alphas, KFold_split)    

minimal_rmse = minimal_mse**(1/2)
   
mse_train, mse_test = train_and_evaluate_ridge(X_train_std, y_train, X_test_std, y_test, optimal_alpha)
rmse_train = mse_train**(1/2)
rmse_test = mse_test**(1/2)

print(f"Train RMSE for daily data: {rmse_train}")
print(f"Test RMSE for daily data: {rmse_test}")
print("")



