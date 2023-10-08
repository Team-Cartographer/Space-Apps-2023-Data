import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from os import path
import data_manager as dm 
import csv
import pickle

def train_xgb_model(X, kp, num_round=500, early_stopping_rounds=None):
    X_train, X_val, kp_train, kp_val = train_test_split(X, kp, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=kp_train)
    dval = xgb.DMatrix(X_val, label=kp_val)

    # XGB Parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # Root Mean Squared Error
        'max_depth': 10,
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


if __name__ == "__main__":
    model = None
    model_path = 'data/models/xgb_model.pkl'
    csv_name = 'data/preds/predictions.csv'

    dataset = dm.get_dataset()[0]
    X = np.array([arr[2] for arr in dataset]).reshape(-1, 1)  # average filtered data (get past reshape issues)
    Y = np.array([arr[0] for arr in dataset])  # kp values

    np.random.shuffle(X)
    np.random.shuffle(Y) 

    print(X.shape, Y.shape)

    # check existence, else 
    if path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            f.close() 
    else:
        model = train_xgb_model(X, Y)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            f.close() 


    print("+------+ Training Complete! +------+")
    print("+--+ Making Predictions +--+")

    # Predict kp values using the trained model
    new_data = np.array([arr[1] for arr in dataset]).reshape(-1, 1)
    
    kp_predictions = predict_kp(model, new_data) 

    with open(csv_name, "w", newline="\n") as f:
        for prediction in kp_predictions:
            f.write(f"{int(np.ceil(prediction))},")

    print(f'Predicted Kp written to {csv_name}".')

