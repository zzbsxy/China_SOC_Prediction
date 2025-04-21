# TabTransformer Feature Importance
def get_tabtransformer_importance(model, X_test, y_test):
    # Extract attention weights from the model or use built-in methods
    feature_importance = model.get_feature_importance()  # The specific method depends on the implementation
    return feature_importance

# Permutation importance (applicable to any model)
def get_permutation_importance(model, X_test, y_test, error_function):
    baseline_error = error_function(y_test, model.predict(X_test))
    importance_scores = {}
    
    for feature in X_test.columns:
        # Save the original data
        X_permuted = X_test.copy()
        # Randomly shuffle the values of a specific feature
        X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
        # Calculate the error after shuffling
        permuted_error = error_function(y_test, model.predict(X_permuted))
        # Importance is the amount of error increase
        importance_scores[feature] = permuted_error - baseline_error
    
    return importance_scores

def calculate_integration_index(model_importances, categorical_features, continuous_features):
    # Normalize all importance scores (make the sum equal to 1)
    total_importance = sum(model_importances.values())
    normalized_importances = {k: v/total_importance for k, v in model_importances.items()}
    
    # Calculate the cumulative importance of different types of features
    cat_importance = sum(normalized_importances[feat] for feat in categorical_features)
    cont_importance = sum(normalized_importances[feat] for feat in continuous_features)
    
    # Calculate the utilization rate of each type of feature (percentage of the total number of features of that type)
    cat_utilization = cat_importance / len(categorical_features)
    cont_utilization = cont_importance / len(continuous_features)
    
    # Feature integration index: balance of utilization between types (close to 1 indicates balanced utilization)
    integration_index = 1 - abs(cat_utilization - cont_utilization)
    
    return integration_index, cat_utilization, cont_utilization