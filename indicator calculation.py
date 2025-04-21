import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Code to load results
df = pd.read_excel("results")

def calculate_metrics(y_true, y_pred):
    r, p = stats.pearsonr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r_squared = r2_score(y_true, y_pred)
    me = np.mean(y_pred - y_true)
    # md = np.mean(y_true - y_pred)
    # Modified MD calculation, applying formula MD = (1/n) × Σ|xi - μ|
    mean_value = np.mean(y_true)  # μ is the mean value of y_true
    md = np.mean(np.abs(y_true - mean_value))  # Calculate the mean of absolute differences between each value and the mean
    mde = np.std(y_true - y_pred, ddof=1)  # Using sample standard deviation
    p_value_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    return {
        "r": r,
        "P-value": p_value_str,
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r_squared,
        "ME": me,
        "MD": md,
        "MDE": mde
    }


training_samples = df[df['Set Type'] == 'Training samples']
testing_samples = df[df['Set Type'] != 'Training samples']

# Calculate metrics for training samples
training_metrics = calculate_metrics(training_samples['Observed SOC (kg C m -2)'],
                                     training_samples['Estimated SOC (kg C m -2)'])

# Calculate metrics for testing samples
testing_metrics = calculate_metrics(testing_samples['Observed SOC (kg C m -2)'],
                                    testing_samples['Estimated SOC (kg C m -2)'])

# Calculate metrics for all samples
all_metrics = calculate_metrics(df['Observed SOC (kg C m -2)'],
                                df['Estimated SOC (kg C m -2)'])

# Output results
print("Training sample metrics:", training_metrics)
print("Testing sample metrics:", testing_metrics)
print("All sample metrics:", all_metrics)