import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from test_bench import date_list
from test_bench import time_list
from test_bench import kp_val
from test_bench import di


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


tot_qdc = quiet_day(date_list, time_list, kp_val)
print(tot_qdc)


def find_kp(x, y, qdc, _c):
    h_val = np.sqrt((x ** 2)+(y ** 2))
    h_max = np.max(h_val)
    a = (h_max - qdc)/qdc
    k_p = 0.67 * np.log10(a) + _c
    return k_p