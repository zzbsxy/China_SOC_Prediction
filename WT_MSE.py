import torch
import torch.nn as nn


class TabTransformerWTLoss(nn.Module):
    """
    Comprehensive loss function for TabTransformer_WT, including weighted MSE for key variables
    and standard MSE for regular variables
    """

    def __init__(self, key_features=None, key_weights=None):
        """
        Initialize the loss function

        Parameters:
        key_features (list): List of names of key features
        key_weights (dict): Dictionary of weights for each key feature
        """
        super(TabTransformerWTLoss, self).__init__()

        # Default values if not provided
        if key_features is None:
            key_features = ['Tmp', 'Srad', 'P/PET', 'NPP']

        if key_weights is None:
            key_weights = {
                'Tmp': 0.1161,
                'Srad': 0.3214,
                'P/PET': 0.3661,
                'NPP': 0.1964
            }

        self.key_features = key_features
        self.key_weights = key_weights
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true, all_features, feature_importance):
        """
        Calculate comprehensive loss

        Parameters:
        y_pred (Tensor): Model's predicted SOC values [batch_size, 1]
        y_true (Tensor): Actual SOC values [batch_size, 1]
        all_features (dict): All input features
        feature_importance (dict): Importance value for each feature

        Returns:
        loss (Tensor): Comprehensive loss value
        """
        # Calculate base MSE (error for each sample)
        base_mse = self.mse(y_pred, y_true)

        # Separate key features and regular features
        key_feature_indices = [i for i, f in enumerate(all_features.keys()) if f in self.key_features]
        other_feature_indices = [i for i, f in enumerate(all_features.keys()) if f not in self.key_features]

        # 1. Apply weighted MSE to key features
        weighted_key_loss = torch.zeros_like(base_mse)
        for feature_name in self.key_features:
            if feature_name in feature_importance:
                weight = self.key_weights[feature_name]
                imp = feature_importance[feature_name]
                weighted_key_loss += weight * imp * base_mse

        # Normalize weighted loss
        total_key_weight = sum(self.key_weights.values())
        weighted_key_loss = weighted_key_loss / total_key_weight

        # 2. Apply standard MSE to regular features
        if other_feature_indices:
            other_features_loss = base_mse  # Regular features use standard MSE
        else:
            other_features_loss = torch.zeros_like(base_mse)

        # 3. Combine the two losses - assuming key features and regular features have equal importance in the total loss
        final_loss = (weighted_key_loss + other_features_loss) / 2

        return final_loss.mean()