import csv 
# import numpy as np
# import os  
# import math 

from utils import * 
from datetime import datetime
from vec3 import *

year, data = 2019, []
data_folder_path: str = "C://Users//ashwa//Desktop//DSCOVR_Data"
data_file_path: str = data_folder_path + f"//dsc_fc_summed_spectra_{year}_vrow1.csv"

# access the data (for now, change this to your local data location) 
with open(data_file_path, mode="r") as data_file:
    data: list = list(csv.reader(data_file, delimiter=','))
    data_file.close()

def get_datetime(row: list) -> str: 
    datetime: str = data[0][0]

def get_mag_field_vec(row: list) -> Vec3:
    mag_field: Vec3 = Vec3(float(data[row][1]), float(data[row][2]), float(data[row][3]))
    return mag_field

def get_flux_measurements(row: list) -> list: 
    flux_measurements: list = [float(data[row][x]) for x in range(4, 53)]
    for i in range(len(flux_measurements)):
        if flux_measurements[i] == 0:
            flux_measurements[i] == "NaN"
        
    return flux_measurements

def get_data_object(row: list, display_flux: bool) -> list:
    data_object = []
    if display_flux:
        data_object = [get_datetime(row), get_mag_field_vec(row), get_flux_measurements(row)] 
    else:
        data_object = [get_datetime(row), get_mag_field_vec(row)]
    
    return data_object

if __name__ == "__main__":
    for row in len(data):
        print(get_data_object(row, display_flux=False))


