import csv 
# import numpy as np
# import os  
# import math 

from utils import * 
from datetime import datetime
from vec3 import *

year = 2019
data_folder_path: str = "C://Users//ashwa//Desktop//DSCOVR_Data"
data_file_path: str = data_folder_path + f"//dsc_fc_summed_spectra_{year}_v01.csv"

# access the data (for now, change this to your local data location) 

with open(data_file_path, mode="r") as data_file:
    data: list = list(csv.reader(data_file, delimiter=','))
    data_file.close()

datetime: str = data[0][0]
mag_field: Vec3 = Vec3(float(data[0][1]), float(data[0][2]), float(data[0][3]))
flux_measurements: list = [float(data[0][x]) for x in range(4, 53)]

data_object = [datetime, mag_field, flux_measurements]

if __name__ == "__main__":
    print(mag_field)


