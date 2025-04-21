import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# Load data
data_0_20cm = pd.read_excel(r"data_path")

# Remove rows with missing values
data_0_20cm_cleaned = data_0_20cm.dropna()

# Perform one-hot encoding for categorical variables
categorical_features = ['Type', 'Soil Type', 'LULC', 'Aspect']
data_0_20cm_encoded = pd.get_dummies(data_0_20cm_cleaned, columns=categorical_features)

# Separate features and target variable
X = data_0_20cm_encoded.drop('SOC (kg C m–2)', axis=1)
y = data_0_20cm_encoded['SOC (kg C m–2)']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train.astype(np.float32))
y_train_torch = torch.tensor(y_train.values.astype(np.float32))
X_test_torch = torch.tensor(X_test.astype(np.float32))
y_test_torch = torch.tensor(y_test.values.astype(np.float32))

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Define RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Initialize hidden state
        h0 = torch.zeros(num_layers, x.size(0), hidden_size)
        # Forward propagation
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Initialize RNN model
input_size = X_train.shape[1]  # Number of features
hidden_size = 10  # Size of hidden layer
num_layers = 2  # Number of RNN layers
output_size = 1  # Output size, predicting SOC

model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100

# Initialize variables to track lowest loss
lowest_train_loss = float('inf')
best_y_train_pred = None
best_y_test_pred = None

# Train the model
model.train()

# for epoch in range(num_epochs):
#     for i, (features, targets) in enumerate(train_loader):
#         # Forward propagation
#         outputs = model(features)
#         loss = criterion(outputs.squeeze(), targets)
#
#         # Backpropagation and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (epoch + 1) % 1 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

for epoch in range(num_epochs):
    for features, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

    # Evaluate model after each epoch
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Make predictions on training set
        y_train_pred_tmp = model(X_train_torch)
        train_loss = criterion(y_train_pred_tmp.squeeze(), y_train_torch)

        # Make predictions on testing set
        y_test_pred_tmp = model(X_test_torch)
        test_loss = criterion(y_test_pred_tmp.squeeze(), y_test_torch)

        # If the training loss for this epoch is the lowest so far, save the prediction results
        if train_loss < lowest_train_loss:
            lowest_train_loss = train_loss.item()
            best_y_train_pred = y_train_pred_tmp.squeeze().cpu().numpy()
            best_y_test_pred = y_test_pred_tmp.squeeze().cpu().numpy()
            print(f'Updated lowest train loss to {lowest_train_loss} at epoch {epoch + 1}')

    model.train()  # Set the model back to training mode

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

print(f'Lowest Train Loss: {lowest_train_loss}')

# Calculate evaluation metrics
# Calculate R^2
r2_train = r2_score(y_train.values, best_y_train_pred)
r2_test = r2_score(y_test.values, best_y_test_pred)
print(f'R^2 (Training): {r2_train:.4f}')
print(f'R^2 (Testing): {r2_test:.4f}')

# Calculate Pearson correlation coefficient
pearson_train, _ = pearsonr(y_train.values, best_y_train_pred)
pearson_test, _ = pearsonr(y_test.values, best_y_test_pred)
print(f'Pearson correlation coefficient (r) (Training): {pearson_train:.4f}')
print(f'Pearson correlation coefficient (r) (Testing): {pearson_test:.4f}')

# Calculate MAE
mae_train = mean_absolute_error(y_train.values, best_y_train_pred)
mae_test = mean_absolute_error(y_test.values, best_y_test_pred)
print(f'Mean Absolute Error (MAE) (Training): {mae_train:.4f}')
print(f'Mean Absolute Error (MAE) (Testing): {mae_test:.4f}')

# Calculate RMSE
rmse_train = np.sqrt(mean_squared_error(y_train.values, best_y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test.values, best_y_test_pred))
print(f'Root Mean Square Error (RMSE) (Training): {rmse_train:.4f}')
print(f'Root Mean Square Error (RMSE) (Testing): {rmse_test:.4f}')
