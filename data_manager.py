import numpy as np
import os  
import math 
import csv
from utils import *


# access the data (for now, change this to your local data location) 
year = 2016
data_folder_path: str = "C://Users//ashwa//Desktop//DSCOVR_Data"
data_file_path: str = data_folder_path + f"//dsc_fc_summed_spectra_{year}_v01.csv"

with open(data_file_path, mode="r") as data_file:
    data = list(csv.reader(data_file, delimiter=','))
    data_file.close()

if __name__ == "__main__":
    print(data[0][0])

mag_field_vector
