# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Load data
data_0_20cm = pd.read_excel(r"data_path")


# Remove rows with missing values
data_0_20cm_cleaned = data_0_20cm.dropna()

# Perform one-hot encoding for categorical variables
categorical_features = ['Type', 'Soil Type', 'LULC', 'Aspect']
data_0_20cm_encoded = pd.get_dummies(data_0_20cm_cleaned, columns=categorical_features)

# Separate features and target variable
X = data_0_20cm_encoded.drop('SOC (kg C m–2)', axis=1)  # Features
y = data_0_20cm_encoded['SOC (kg C m–2)']               # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter search space
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Categorical(['auto', 'sqrt']),
    'bootstrap': Categorical([True, False])
}

# Use Bayesian optimization to search for optimal parameters
# Note: BayesSearchCV will still use the default cross-validation strategy internally
bayes_search = BayesSearchCV(
    RandomForestRegressor(random_state=42),
    param_space,
    n_iter=50,               # Number of Bayesian optimization iterations
    n_jobs=-1,               # Use all available CPUs
    random_state=42,
    scoring='neg_mean_squared_error',
    verbose=1
)

# Fit the model
bayes_search.fit(X_train, y_train)

# Output best parameters and score
print(f"Best parameters: {bayes_search.best_params_}")
print(f"Best Bayesian search score: {-bayes_search.best_score_}")

# Make predictions using the best model
best_rf = bayes_search.best_estimator_
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

# Calculate various metrics
# Calculate R^2
r2 = r2_score(y_test, y_test_pred)
print(f'R^2: {r2}')

# Calculate Pearson correlation coefficient
pearson_corr, _ = pearsonr(y_test, y_test_pred)
print(f'Pearson correlation coefficient(r): {pearson_corr}')

# Calculate MAE
mae = mean_absolute_error(y_test, y_test_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'Root Mean Square Error (RMSE): {rmse}')

