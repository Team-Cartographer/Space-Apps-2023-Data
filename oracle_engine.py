import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from test_bench import date_list
from test_bench import time_list
from test_bench import kp_val
from test_bench import di


print(len(date_list))
print(len(time_list))
print(len(kp_val))
def quiet_day(date, time, kp_list):
    sum = 0
    itms = 0
    avg = 0
    threshold = 2
    for i in kp_list:
        if i <= 2:
            sum += i
            itms += 1
            # print(date[i])
    k_avg = sum/itms

    dictframe = pd.DataFrame({'day': date, 'time': time, 'kp': kp_list})
    dictframe['day'] = pd.to_datetime(dictframe['day'])
    filtered_days = dictframe.groupby('day')['kp'].apply(lambda x: (x > threshold).any())
    quiet_day_list = filtered_days[filtered_days == False].index.tolist()

    # print(quiet_day_list)

    return quiet_day_list

tot_qdc = quiet_day(date_list, time_list, kp_val)
print(tot_qdc)

def find_kp(x, y, qdc, _C):
    h_val = np.sqrt((x ** 2)+(y ** 2))
    h_max = np.max(h_val)
    a = (h_max - qdc)/qdc
    k_p = 0.67 * np.log10(a) + _C
    return k_p