from utils import *
# code provided from nasa

import pandas
data = pandas.read_csv("dsc_fc_summed_spectra_2016_v01.csv", delimiter = ',', parse_dates=[0],
                       infer_datetime_format=True, na_values='0', header = None)
