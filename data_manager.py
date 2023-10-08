import numpy as np
import csv 
from datetime import datetime, timedelta
from utils import * 
from tqdm import tqdm 
from backup_engine import KalmanFilter3D

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
    mag_field = np.array([float(data[row][1]), float(data[row][2]), float(data[row][3])])
    return mag_field

def get_mag_vec_from_list(lis: list):
    mag_field = np.array([float(lis[1]), float(lis[2]), float(lis[3])])
    return mag_field

def get_flux_measurements(row: int) -> list:
    flux_measurements: list = [float(data[row][x]) for x in range(4, 53)]
    flux_measurements = ["NaN" if measurement == 0 else measurement for measurement in flux_measurements]

    return flux_measurements

def get_flux_from_list(lis: list) -> list:
    flux_measurements: list = [float(lis[x]) for x in range(4, 53)]
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

class DataFrame:
    def __init__(self, data_row):
        self.date = self.process_date(data_row[0]) 
        self.plus3hours = self.date + timedelta(hours=3)
        self.dates = [self.date]

        self.raw_vectors, self.faraday_readings = self.fill_raws()
        self.filtered_vectors = []

        self.init_vec = self.raw_vectors[0]
        kalman_filter = KalmanFilter3D([self.init_vec[0], self.init_vec[1], self.init_vec[2]])
        self.filtered_vectors = kalman_filter.filter_measurements(self.raw_vectors)
        # len(filtered_vectors) == len(raw_vectors) == 181 
        
        self.obj = [self.date, self.plus3hours, self.raw_vectors, self.filtered_vectors, self.faraday_readings]

    
    def process_date(self, date_string): 
        date_format = "%Y-%m-%d %H:%M:%S"
        date_object = datetime.strptime(date_string, date_format)
        return date_object
    
    def fill_raws(self):
        raw_vecs = []
        farad_r = []
        for row in data:
            date = self.process_date(row[0])
            if(self.date <= date <= self.plus3hours):
                self.dates.append(date)
                vecs = np.array(get_mag_vec_from_list(row))
                raw_vecs.append(vecs)
                farad_r = get_flux_from_list(row)
        
        self.dates.append(self.plus3hours)
        
        return raw_vecs, farad_r
    
    def get_data_frame(self): 
        return self.obj
    
    def show_data(self):
        import matplotlib.pyplot as plt 
        fig, ax1 = plt.subplots()
        x = self.dates[1:len(self.dates)-1]

        # Plot the first data on the first Y-axis (left)
        ax1.plot(x, self.raw_vectors, color='tab:gray')
        ax1.set_xlabel('3 Hours')
        ax1.set_ylabel('Raw Vectors', color='tab:gray')

        # Create a second set of Y-axes that shares the same X-axis
        ax2 = ax1.twinx()

        # Plot the second data on the second Y-axis (right)
        ax2.plot(x, self.filtered_vectors, color='000000')
        ax2.set_ylabel('Filtered Vectors', color='000000')

        # Add a title
        plt.title(f'Vectors for 3 Hours from {self.date} to {self.plus3hours}')

        # Show the plot
        plt.show()


if __name__ == "__main__":
    # in the end, this should gain data from a live source, and then push it out to firebase. 
    # this file will probably require some changes soon 

    df = DataFrame(get_data_list(True)[180]) ### every 180 steps you move forward 3 hours ###
    df.show_data()

    

    
