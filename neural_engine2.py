import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_xgb_model(X_train, y_train, num_round=100, early_stopping_rounds=None):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Create a DMatrix for efficient handling of data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # Root Mean Squared Error
        'max_depth': 6,
        'eta': 0.3,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train the XGBoost model
    evals = [(dtrain, 'train'), (dval, 'eval')]
    xgb_model = xgb.train(params, dtrain, num_round, evals=evals, early_stopping_rounds=early_stopping_rounds)

    return xgb_model

def predict_kp(xgb_model, X_test):
    dtest = xgb.DMatrix(X_test)
    kp_predictions = xgb_model.predict(dtest)
    return kp_predictions

# Example usage:
if __name__ == "__main__":
    
    np.random.seed(42)
    num_samples = 1000
    num_features = 10
    X = np.random.rand(num_samples, num_features)
    kp = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(num_samples)  # Example kp calculation

    # Train the XGBoost model
    xgb_model = train_xgb_model(X, kp, num_round=100, early_stopping_rounds=10)

    # Predict kp values using the trained model
    new_data = np.random.rand(10, num_features)  # New input data
    kp_predictions = predict_kp(xgb_model, new_data)

    print("Predicted kp values:", kp_predictions)