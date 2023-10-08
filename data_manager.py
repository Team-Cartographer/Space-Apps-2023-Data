import numpy as np
import csv 
from datetime import datetime, timedelta
from utils import * 
from tqdm import tqdm 
from backup_engine import KalmanFilter3D

@timeit
def setup_data(data) -> np.ndarray:
    dtype = [('datetime', 'U19'), ('mag_field', '3f8'), ('flux_measurements', '49f8')]

    data_array = np.empty(len(data), dtype=dtype)

    for i in tqdm(range(len(data)), desc="Compiling Data"):
        datetime = data[i][0]
        mag_field = np.array([float(data[i][1]), float(data[i][2]), float(data[i][3])], dtype=np.float64)
        flux_measurements = [float(data[i][x]) for x in range(4, 53)]
        flux_measurements = [np.nan if measurement == 0 else measurement for measurement in flux_measurements]

        row_data = (datetime, mag_field, flux_measurements)

        data_array[i] = row_data

    return data_array


@timeit
def create_dataset(data, kpData):
    first_reading = data['mag_field'][0]
    filt = KalmanFilter3D([first_reading[0], first_reading[1], first_reading[2]])
    filtered_data = filt.filter_measurements(data['mag_field'])

    dataset = []
    temp_row = []
    filtered_avg = []

    for i in tqdm(range(len(data)), desc="Building Dataset"): 
        if i % 180 == 0: 
            # 3 hour gap every 180 points  
            #print(f"[{temp_row[0]}, {temp_row[1][0:10]}]\n") if i != 0 else None
            if i != 0:
                average = np.mean(np.mean(filtered_avg, axis=1))
                temp_row.append(average)
            dataset.append(temp_row) # if i != 0 else None 
            temp_row = []

            start_time = data['datetime'][i]
            start_date = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
            #end_time = (start_date + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
            kp_value = int(kpData[start_date.strftime("%Y-%m-%d %H:%M:%S")])

            #temp_row.append(start_time)
            #temp_row.append(end_time)
            #temp_row.append([]) # raw mag field
            #temp_row.append([]) # faraday data

            temp_row.append(kp_value)
            temp_row.append([]) # filtered mag field 
        else: 
            filtered_avg.append(filtered_data[i])
            temp_row[1].append(filtered_data[i])
            #temp_row[2].append(data['mag_field'][i])
            #temp_row[3].append(data['flux_measurements'][i])

    return dataset[1:]


def get_training_data(start_year=2016, years=1, print_year=False):
    if years > 5 or start_year > 2021:
        raise ValueError # years cannot be greater than 5 until alex finishes kp data calculation 
    
    rows, all_datasets = [], []
    KpDict: dict = {}
    with open('K_p_DATA.dat', mode='r') as f:
        for line in f:
            fields = line.strip().split()
            rows.append(fields)
            frmt = "%Y-%m-%d %H:%M:%S"
            timestr = fields[0] + " " + fields[1]
            timestamp = datetime.strptime(timestr[0:len(timestr) - 4], frmt)
            KpDict.update({str(timestamp): int(fields[3][0:1])})
        
        f.close()
    

    for i in range(0, years): 
        year = start_year + i # DO NOT USE >2022 YET
        print(f'{i+1}. {year}') if print_year else None
        # customize to your liking 
        data_folder_path: str = "C://Users//ashwa//Desktop//DSCOVR_Data" 
        data_file_path: str = data_folder_path + f"//dsc_fc_summed_spectra_{year}_v01.csv"
        with open(data_file_path, mode="r") as data_file:
            data: list = list(csv.reader(data_file, delimiter=','))
            data_file.close()

        processed_data = setup_data(data)
        cleaned_data = create_dataset(processed_data, KpDict)
        all_datasets.append(cleaned_data)
    
    return all_datasets


if __name__ == "__main__":
    # in the end, this should gain data from a live source, and then push it out to mongo. 
    # this file will probably require some changes soon 

    # Example usage
    #x = get_training_data(1)[0][i][1][1] -> filtered data per 3 hour period 
    # print(x)
    # indexes, explained -> 
    # first [0] -> all_datasets[0] = dataset for 2016 (first year)
    # second [0] -> first entry dataset for 2016 [constant_row, training_row]
    # third [1] -> training_row from first dataset for 2016
    # fourth [0] -> kp value fromt training row for first dataset (2016) [1] would be the filtered data
    dataset = get_training_data()[0]
    print(np.array([arr[1] for arr in dataset]).reshape(-1, 1))
    
    pass
