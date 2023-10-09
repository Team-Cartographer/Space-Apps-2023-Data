import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from os import path
import data_manager as dm 
import pickle
from tqdm import tqdm 
from random import choice 

def train_xgb_model(X, kp, num_round=750, early_stopping_rounds=None):
    X_train, X_val, kp_train, kp_val = train_test_split(X, kp, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=kp_train)
    dval = xgb.DMatrix(X_val, label=kp_val)

    # XGB Parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # Root Mean Squared Error
        'max_depth': 35,
        'eta': 0.2,
        'gamma': 0.4,
        'subsample': 0.8,
        'colsample_bytree': 0.6
    }

    # Train the XGBoost model
    evals = [(dtrain, 'train'), (dval, 'eval')]
    xgb_model = xgb.train(
        params, dtrain, num_round, evals=evals, early_stopping_rounds=early_stopping_rounds,
        )

    return xgb_model


def predict_kp(xgb_model, X_test):
    dtest = xgb.DMatrix(X_test)
    kp_predictions = xgb_model.predict(dtest)
    return kp_predictions


if __name__ == "__main__":
    model = None
    model_path = 'data/models/xgb_model.pkl'
    csv_name = 'data/preds/predictions.csv'

    dataset = dm.get_dataset(start_year=2017)[0]
    X = np.array([arr[2] for arr in dataset]).reshape(-1, 1)  # average filtered data (get past reshape issues)
    Y = np.array([arr[0] for arr in dataset])  # kp values

    #np.random.shuffle(X)
    #np.random.shuffle(Y) 
    #print(X.shape, Y.shape)

    # check existence, else 
    if path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            f.close() 
    else:
        model = train_xgb_model(X, Y)
        print("+------+ Training Complete! +------+")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            f.close() 

    print("+------+ Making Predictions +------+")

    # Predict kp values using the trained model
    new_data = np.array([arr[1] for arr in dataset]).reshape(-1, 1)
    
    kp_predictions = predict_kp(model, new_data)
    num_groups = 2920

    # Calculate the approximate group size
    group_size = len(kp_predictions) // num_groups

    # Calculate the number of elements that won't fit into groups evenly
    remaining_elements = len(kp_predictions) % num_groups

    # Initialize the start and end indices for slicing
    start_idx = 0
    groups = []

    # Create the groups
    for i in tqdm(range(num_groups), desc="grouping"):
        # Calculate the end index for slicing
        end_idx = start_idx + group_size + (1 if i < remaining_elements else 0)
        
        # Slice the array to create a group
        group = kp_predictions[start_idx:end_idx]
        
        # Add the group to the list of groups
        groups.append(group)
        
        # Update the start index for the next iteration
        start_idx = end_idx
    
    averages = []
    for group in tqdm(groups, desc="averaging"):
        averages.append(choice(list(set(group))))

    with open(csv_name, "w", newline="\n") as f:
        for avg in averages:
            f.write(f"{int(np.floor(avg))},")

    print(len(averages))

    print(f'Predicted Kp written to {csv_name}".')

