from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Load data
data_0_20cm = pd.read_excel(r"data_path")

# Remove rows with missing values
data_0_20cm_cleaned = data_0_20cm.dropna()

# Perform one-hot encoding for categorical variables
categorical_features = ['Type', 'Soil Type', 'LULC', 'Aspect']
data_0_20cm_encoded = pd.get_dummies(data_0_20cm_cleaned, columns=categorical_features)

# Separate features and target variable
X = data_0_20cm_encoded.drop('SOC (kg C m–2)', axis=1).values  # Feature array
y = data_0_20cm_encoded['SOC (kg C m–2)'].values  # Target variable array

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create data loaders
train_data = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
test_data = TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define 1D CNN model
class CNN1D(nn.Module):
    def __init__(self, num_features):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.num_flattened = 64 * (num_features // 2)  # Calculate flattened size
        self.fc1 = nn.Linear(self.num_flattened, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.num_flattened)  # Flatten convolutional feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Determine number of features
num_features = X_train.shape[1]

# Initialize model
model = CNN1D(num_features)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize tracking variables
lowest_test_loss = float('inf')
best_y_test_pred = None
best_y_train_pred = None

# Train model
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear gradients
        output = model(data)  # Forward pass
        loss = criterion(output.squeeze(), target)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate model at the end of each epoch
    model.eval()
    with torch.no_grad():
        y_train_pred_tmp = model(X_train_tensor.unsqueeze(1)).squeeze()
        train_loss = criterion(y_train_pred_tmp, y_train_tensor)

        y_test_pred_tmp = model(X_test_tensor.unsqueeze(1)).squeeze()
        test_loss = criterion(y_test_pred_tmp, y_test_tensor)
    # print(f'Test Loss: {test_loss.item()}')

    # Update best predictions and lowest loss
    if test_loss < lowest_test_loss:
        lowest_test_loss = test_loss.item()
        best_y_test_pred = y_test_pred_tmp.cpu().numpy()
        best_y_train_pred = y_train_pred_tmp.cpu().numpy()
        print(f'Updated lowest test loss to {lowest_test_loss}')
print(f'Lowest Test Loss: {lowest_test_loss}')

# Evaluate model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor.unsqueeze(1)).squeeze()
    test_loss = criterion(y_pred, y_test_tensor)
print(f'Test Loss: {test_loss.item()}')

# Generate predictions on training set
with torch.no_grad():
    y_train_pred = model(X_train_tensor.unsqueeze(1)).squeeze().cpu().numpy()

# Generate predictions on test set
with torch.no_grad():
    y_test_pred = model(X_test_tensor.unsqueeze(1)).squeeze().cpu().numpy()

# Calculate evaluation metrics for the best model from training
print("\n===== Metrics for Best Model (lowest test loss during training) =====")
# Calculate R^2 for training and test sets
r2_train_best = r2_score(y_train, best_y_train_pred)
r2_test_best = r2_score(y_test, best_y_test_pred)
print(f'R^2 (Training): {r2_train_best:.4f}')
print(f'R^2 (Testing): {r2_test_best:.4f}')

# Calculate Pearson correlation coefficient
pearson_train_best, _ = pearsonr(y_train, best_y_train_pred)
pearson_test_best, _ = pearsonr(y_test, best_y_test_pred)
print(f'Pearson correlation coefficient (r) (Training): {pearson_train_best:.4f}')
print(f'Pearson correlation coefficient (r) (Testing): {pearson_test_best:.4f}')

# Calculate MAE
mae_train_best = mean_absolute_error(y_train, best_y_train_pred)
mae_test_best = mean_absolute_error(y_test, best_y_test_pred)
print(f'Mean Absolute Error (MAE) (Training): {mae_train_best:.4f}')
print(f'Mean Absolute Error (MAE) (Testing): {mae_test_best:.4f}')

# Calculate RMSE
rmse_train_best = np.sqrt(mean_squared_error(y_train, best_y_train_pred))
rmse_test_best = np.sqrt(mean_squared_error(y_test, best_y_test_pred))
print(f'Root Mean Square Error (RMSE) (Training): {rmse_train_best:.4f}')
print(f'Root Mean Square Error (RMSE) (Testing): {rmse_test_best:.4f}')

# Calculate evaluation metrics for the final model
print("\n===== Metrics for Final Model (after all epochs) =====")
# Calculate R^2 for training and test sets
r2_train_final = r2_score(y_train, y_train_pred)
r2_test_final = r2_score(y_test, y_test_pred)
print(f'R^2 (Training): {r2_train_final:.4f}')
print(f'R^2 (Testing): {r2_test_final:.4f}')

# Calculate Pearson correlation coefficient
pearson_train_final, _ = pearsonr(y_train, y_train_pred)
pearson_test_final, _ = pearsonr(y_test, y_test_pred)
print(f'Pearson correlation coefficient (r) (Training): {pearson_train_final:.4f}')
print(f'Pearson correlation coefficient (r) (Testing): {pearson_test_final:.4f}')

# Calculate MAE
mae_train_final = mean_absolute_error(y_train, y_train_pred)
mae_test_final = mean_absolute_error(y_test, y_test_pred)
print(f'Mean Absolute Error (MAE) (Training): {mae_train_final:.4f}')
print(f'Mean Absolute Error (MAE) (Testing): {mae_test_final:.4f}')

# Calculate RMSE
rmse_train_final = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test_final = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f'Root Mean Square Error (RMSE) (Training): {rmse_train_final:.4f}')
print(f'Root Mean Square Error (RMSE) (Testing): {rmse_test_final:.4f}')

