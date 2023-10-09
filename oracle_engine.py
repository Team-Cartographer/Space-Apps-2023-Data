import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
from test_bench import date_list
from test_bench import time_list
from test_bench import kp_val
from test_bench import di
import data_manager as dm


## print(len(date_list))
# print(len(time_list))
# print(len(kp_val))


def quiet_day(date, time, kp_list):
    sum = 0
    itms = 0
    avg = 0
    threshold = 2

    # getting list of quiet days
    dictframe = pd.DataFrame({'day': date, 'time': time, 'kp': kp_list})
    # dictframe['day'] = pd.to_datetime(dictframe['day'])
    filtered_days = dictframe.groupby('day')['kp'].apply(lambda x: (x > threshold).any())
    quiet_day_list = filtered_days[filtered_days == False].index.tolist()
    # print(quiet_day_list)

    # getting the average kp value for those days
    data = {
        'date': date_list,
        'time': time_list,
        'Kp': kp_list
    }
    df = pd.DataFrame(data)
    # print(data)
    datetime_list = (df['date'])
    filtered_df = df[df['date'].isin(quiet_day_list)]
    # print(filtered_df)
    avg_kp = filtered_df.groupby('date')['Kp'].mean()

    # overall averge mean of kp value on quiet days
    ovr_avg_kp = avg_kp.mean()
    sum_quotients = 0
    # now normalizing the data and averaging to finally get QDC
    for i in kp_list:
        quotient = int(i)/ovr_avg_kp
        sum_quotients += quotient

    qdc = sum_quotients/len(kp_list)

    # print(ovr_avg_kp)

    return qdc

rows = []
interval = 10
def find_kp(qdc):
    # All that needs to be done is to put the filtered x and y into this algorithm
    data = dm.get_dataset()[0]
    filtered = np.array([arr[2] for arr in data])
    h_vals = []
    for i in filtered:
        h_vals.append(np.sqrt((i[0] ** 2) + (i[1] ** 2)))

    h_max = np.max(h_vals)
    a = (h_max - qdc)/qdc
    k_p = 0.67 * np.log10(a)
    return k_p


def predict_kp(init_kp, kp_list):
    initial_state_covariance = np.array([1.0])
    initial_state = np.array([init_kp])

    A = np.array([1.0])
    B = np.array([1.0])
    H = np.array([1.0])

    Q = np.array([0.01])
    R = np.array([0.1])

    state_estimate = initial_state
    state_covariance = initial_state_covariance

    for kp in tqdm(kp_list[len(kp_list)-20:], desc="Predicting State"):
        # prediction step
        predicted_kp = np.dot(A, kp)
        predicted_state_covariance = np.dot(A, state_covariance)

        # update step
        kalman_gain = predicted_state_covariance / (predicted_state_covariance + R)
        state_estimate = predicted_kp + kalman_gain * (kp_list - np.dot(H, predicted_kp))
        state_covariance = (1-kalman_gain) * predicted_state_covariance

        #print("estimated state:", state_estimate[0])

    return int(predicted_kp[0])

if __name__ == '__main__':
    tot_qdc = quiet_day(date_list, time_list, kp_val)
    kp_data = []
    YEAR = 2022

    kp_data_2022 = []
    with open('data//preds//2022_predictions.csv', 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split(',')
            values = values[0:len(values) -1]
            kp_data_2022.extend([int(value) for value in values])
    print(len(kp_data_2022))
    print(kp_data_2022)
    
    # kp_data = np.array(kp_data)
    # init_kp = kp_data[len(kp_data)-20]
    #
    # print(predict_kp(init_kp, kp_data)) # +-15% error on the predicted value