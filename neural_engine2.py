import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import data_manager as dm 
import csv
import pickle


def train_xgb_model(X, kp, num_round=500, early_stopping_rounds=None):
    # Split the data into training and validation sets
    X_train, X_val, kp_train, kp_val = train_test_split(X, kp, test_size=0.2, random_state=42)

    # Create DMatrix for efficient handling of data
    dtrain = xgb.DMatrix(X_train, label=kp_train)
    dval = xgb.DMatrix(X_val, label=kp_val)

    # Define XGBoost parameters
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

def train_xgb():
    xgb_model = train_xgb_model(X, Y, num_round=100,
                                early_stopping_rounds=10)  # TODO JL setup a pickle file for this thing

    # Pickles xgb_model
    with open('xgb_model.pickle', 'wb') as model_file:
        pickle.dump(xgb_model, model_file)

    return xgb_model


if __name__ == "__main__":
    dataset = dm.get_training_data()[0]
    X = np.array([arr[2] for arr in dataset]).reshape(-1, 1)  # average filtered data (get past reshape issues)
    Y = np.array([arr[0] for arr in dataset])  # kp values
    print(X.shape, Y.shape)

    # Train the XGBoost model
    train_xgb()

    print("+------+ Training Complete! +------+")

    # Predict kp values using the trained model
    new_data = np.array([arr[1] for arr in dataset]).reshape(-1, 1)
    try:
        with open('xgb_model.pickle', 'rb') as model_file:
            loaded_xgb_model = pickle.load(model_file)
            kp_predictions = predict_kp(loaded_xgb_model, new_data)
    except e:
        kp_predictions = predict_kp(train_xgb_model(), new_data)  # TODO check for xgb model via pickle before creating a new one

    print("Predicted Kp Values: ", kp_predictions)

    # TODO write kp_predictions to a CSV
    with open("predictions.csv", "w", newline="\n") as file:
        writer = csv.writer(file, delimiter=',')
        for row in kp_predictions:
            csv_writer.writerow(row)

    print(f'Data from kp_predictions has been written to {csv_file_name}.')
