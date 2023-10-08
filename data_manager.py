import csv 
import numpy as np

from datetime import datetime, timedelta
from utils import * 
from tqdm import tqdm 

year = 2016
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
    mag_field = [float(data[row][1]), float(data[row][2]), float(data[row][3])]
    return mag_field


def get_flux_measurements(row: int) -> list:
    flux_measurements: list = [float(data[row][x]) for x in range(4, 53)]
    flux_measurements = ["NaN" if measurement == 0 else measurement for measurement in flux_measurements]

    return flux_measurements


def get_data_object(row: int, display_flux: bool) -> list:  # add additional proper type hinting later
    data_object: list  # add additional typing later
    if display_flux:
        data_object = [get_datetime(row), get_mag_field_vec(row), get_flux_measurements(row)] 
    else:
        data_object = [get_datetime(row), get_mag_field_vec(row)]
    
    return data_object

def get_data_list(disp_flux: bool) -> list:
    data_list = []
    for i in tqdm(range(len(data)), desc="Compiling Data"): 
        data_list.append(get_data_object(i, display_flux=disp_flux))
    
    return data_list

# class DataFrame:
#     def __init__(self, data_row):
#         self.date = self.process_date(data_row[0]) 
#         self.plus3hours = self.date + timedelta(hours=3)

#         self.raw_vectors = []
#         self.filtered_vectors = []
    
#     def process_date(self, date_string): 
#         date_format = "%Y-%m-%d %H:%M:%S"
#         date_object = datetime.strptime(date_string, date_format)
#         return date_object
    
#     def fill_raw_vec(self):
#         for row in data:
            
    




if __name__ == "__main__":
    # in the end, this should gain data from a live source, and then push it out to firebase. 
    # this file will probably require some changes soon 

    # print(get_data_list(False)[12600:12700])
    df = DataFrame(get_data_list(False)[0])
    print(df.date)
    print(df.plus3hours)


