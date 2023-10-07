import csv 
# import numpy as np
# import os  
# import math 

from utils import * 
from datetime import datetime
from tqdm import tqdm 

year = 2019
data_folder_path: str = "C://Users//ashwa//Desktop//DSCOVR_Data"
data_file_path: str = data_folder_path + f"//dsc_fc_summed_spectra_{year}_v01.csv"

# access the data (for now, change this to your local data location) 
with open(data_file_path, mode="r") as data_file:
    data: list = list(csv.reader(data_file, delimiter=','))
    data_file.close()

def get_datetime(row: int) -> str: 
    datetime: str = data[row][0]
    return datetime

def get_mag_field_vec(row: int):
    mag_field = (float(data[row][1]), float(data[row][2]), float(data[row][3]))
    return mag_field

def get_flux_measurements(row: int) -> list: 
    flux_measurements: list = [float(data[row][x]) for x in range(4, 53)]
    for i in range(len(flux_measurements)):
        if flux_measurements[i] == 0:
            flux_measurements[i] == "NaN"
        
    return flux_measurements

def get_data_object(row: int, display_flux: bool) -> list:
    data_object = []
    if display_flux:
        data_object = [get_datetime(row), get_mag_field_vec(row), get_flux_measurements(row)] 
    else:
        data_object = [get_datetime(row), get_mag_field_vec(row)]
    
    return data_object

if __name__ == "__main__":
    data_list = []
    for i in tqdm(range(len(data)), desc="Compiling Data"): 
        data_list.append(get_data_object(i, display_flux=False))
    
    # not working, but leave it for now 
    # x = input(f"Enter a date in '{year}-MM-DD' format: ")
    # for daily_data in data_list:
    #     if(x == daily_data[0].split()[0]):
    #         print(daily_data)


