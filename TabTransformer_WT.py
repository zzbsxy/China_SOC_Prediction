import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
import sys
from pathlib import Path

# Import the custom loss function
# Assuming WT_MSE.py is in the same directory
sys.path.append(str(Path.cwd()))
from WT_MSE import TabTransformerWTLoss

# Set default tensor type to Double
torch.set_default_tensor_type(torch.DoubleTensor)

# Read data
data = pd.read_excel("data_path.xlsx")  # Replace with actual data path

# Separate features and target variable
X = data.drop('SOC (kg C m–2)', axis=1)
y = data['SOC (kg C m–2)']

# Convert categorical features to numerical encoding
cat_features = ['Type', 'Soil Type', 'Aspect', 'LULC']  # Modify according to data
for col in cat_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize continuous features
cont_features = X_train.columns.drop(cat_features)  # List of continuous features
scaler = StandardScaler()
X_train[cont_features] = scaler.fit_transform(X_train[cont_features])
X_test[cont_features] = scaler.transform(X_test[cont_features])

# Prepare TabTransformer input
cat_dims = [int(X_train[col].nunique()) for col in cat_features]  # Number of unique values for each categorical feature
cont_mean_std = np.array(
    [[X_train[col].mean(), X_train[col].std()] for col in cont_features])  # Mean and std of continuous features
cat_dims_updated = [X[col].nunique() for col in cat_features]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define TabTransformer model
model = TabTransformer(
    categories=cat_dims_updated,
    num_continuous=len(cont_features),
    dim=32,
    dim_out=1,
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
    continuous_mean_std=torch.tensor(cont_mean_std)
)

# Prepare training data
X_train_cat = torch.tensor(X_train[cat_features].values).to(device)
X_train_cont = torch.tensor(X_train[cont_features].values).to(device)
y_train = torch.tensor(y_train.values).unsqueeze(1).to(device)

# Create a feature to index mapping for tracking importance
feature_to_idx = {feat: i for i, feat in enumerate(cont_features)}
for i, feat in enumerate(cat_features):
    feature_to_idx[feat] = i + len(cont_features)

# Initialize custom loss function
criterion = TabTransformerWTLoss()  # Now using the class with default values

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Define a function to compute feature importance (simplified)
def compute_feature_importance(model, X_cat, X_cont):
    # This is a placeholder - in a real implementation, you'd use a method like
    # permutation importance, SHAP values, or the model's attention weights
    # Here we'll randomly generate importance for demonstration
    all_features = list(cont_features) + list(cat_features)
    importance = {feat: np.random.random() for feat in all_features}
    # Normalize to sum to 1
    total_imp = sum(importance.values())
    importance = {k: v / total_imp for k, v in importance.items()}
    return importance


# Train model
model = model.to(device)
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(X_train_cat, X_train_cont)

    # Compute feature importance
    feature_importance = compute_feature_importance(model, X_train_cat, X_train_cont)

    # Create features dictionary for loss function
    features_dict = {}
    for i, col in enumerate(cont_features):
        features_dict[col] = X_train_cont[:, i]
    for i, col in enumerate(cat_features):
        features_dict[col] = X_train_cat[:, i]

    # Compute loss
    loss = criterion(y_pred, y_train, features_dict, feature_importance)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

# Evaluate model on test set
X_test_cat = torch.tensor(X_test[cat_features].values).to(device)
X_test_cont = torch.tensor(X_test[cont_features].values).to(device)

model.eval()
with torch.no_grad():
    y_pred = model(X_test_cat, X_test_cont).cpu().numpy()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {mse:.4f}, Test R2: {r2:.4f}')
