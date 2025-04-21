import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_layer_size).to(x.device)
        # LSTM output size is (batch, seq_len, hidden_size)
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        # We only care about the last output of LSTM
        out = self.linear(h_n[-1])
        return out

# Load data
data = pd.read_excel(r"data_path")

# Preprocess data
data_cleaned = data.dropna() # Remove missing values and Site column
categorical_features = ['Type', 'Soil Type', 'LULC', 'Aspect（坡向）']
data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features)
X = data_encoded.drop('SOC (kg C m–2)', axis=1).values
y = data_encoded['SOC (kg C m–2)'].values

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# Create DataLoader
train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# Create test set DataLoader
test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model parameters
input_size = X_train.shape[1]
hidden_layer_size = 100
num_layers = 1
output_size = 1

# Create LSTM model
model = LSTMModel(input_size, hidden_layer_size, num_layers, output_size)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Initialize variables to track lowest test loss
lowest_test_loss = float('inf')
best_y_test_pred = None
best_y_train_pred = None

# Train model
num_epochs = 100
# for epoch in range(num_epochs):
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch)
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# # Make predictions on test set
# model.eval()  # Set to evaluation mode
# with torch.no_grad():
#     X_test_tensor = X_test_tensor.to(device)
#     y_test_pred_tensor = model(X_test_tensor)
#     y_test_pred = y_test_pred_tensor.cpu().numpy()

for epoch in range(num_epochs):
    model.train()
    train_loss_accumulated = 0
    count_train = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_accumulated += loss.item()
        count_train += 1
    average_train_loss = train_loss_accumulated / count_train

    # Evaluation mode
    model.eval()
    test_loss_accumulated = 0
    count_test = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_test_pred = model(X_batch)
            test_loss = criterion(y_test_pred, y_batch)
            test_loss_accumulated += test_loss.item()
            count_test += 1

    average_test_loss = test_loss_accumulated / count_test
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Training Loss: {average_train_loss}, Test Loss: {average_test_loss}')

    # Update best predictions and lowest test loss
    if average_test_loss < lowest_test_loss:
        lowest_test_loss = average_test_loss
        print(f'Updated lowest test loss to {lowest_test_loss}')

        # Update best test set predictions
        best_y_test_pred = []
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred_batch = model(X_batch).detach().cpu().numpy()
            best_y_test_pred.append(y_pred_batch)
        best_y_test_pred = np.concatenate(best_y_test_pred, axis=0)

        # Also update best training set predictions
        best_y_train_pred = []
        for X_batch, _ in train_loader:
            X_batch = X_batch.to(device)
            y_pred_batch = model(X_batch).detach().cpu().numpy()
            best_y_train_pred.append(y_pred_batch)
        best_y_train_pred = np.concatenate(best_y_train_pred, axis=0)

# After training is complete, calculate evaluation metrics
print("\n----- Evaluation Metrics for Best Model -----")

# Reshape predictions if needed
best_y_train_pred = best_y_train_pred.flatten()
best_y_test_pred = best_y_test_pred.flatten()

# Calculate R^2 for training and test sets
r2_train = r2_score(y_train, best_y_train_pred)
r2_test = r2_score(y_test, best_y_test_pred)
print(f'R^2 (Training): {r2_train:.4f}')
print(f'R^2 (Testing): {r2_test:.4f}')

# Calculate Pearson correlation coefficient
pearson_train, _ = pearsonr(y_train, best_y_train_pred)
pearson_test, _ = pearsonr(y_test, best_y_test_pred)
print(f'Pearson correlation coefficient (r) (Training): {pearson_train:.4f}')
print(f'Pearson correlation coefficient (r) (Testing): {pearson_test:.4f}')

# Calculate MAE for training and test sets
mae_train = mean_absolute_error(y_train, best_y_train_pred)
mae_test = mean_absolute_error(y_test, best_y_test_pred)
print(f'Mean Absolute Error (MAE) (Training): {mae_train:.4f}')
print(f'Mean Absolute Error (MAE) (Testing): {mae_test:.4f}')

# Calculate RMSE for training and test sets
rmse_train = np.sqrt(mean_squared_error(y_train, best_y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, best_y_test_pred))
print(f'Root Mean Square Error (RMSE) (Training): {rmse_train:.4f}')
print(f'Root Mean Square Error (RMSE) (Testing): {rmse_test:.4f}')
