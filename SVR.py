from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

# Initialize and train Support Vector Regression model
# Use StandardScaler for feature scaling, as SVR is very sensitive to feature scaling
svr_regressor = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))

svr_regressor.fit(X_train, y_train)

# Make predictions on training and testing sets
y_train_pred = svr_regressor.predict(X_train)
y_test_pred = svr_regressor.predict(X_test)



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