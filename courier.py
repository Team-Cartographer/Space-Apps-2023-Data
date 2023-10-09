from datetime import datetime, timedelta
import json 

YEAR = 2016

kp_data = []
with open(f'data//preds//{YEAR}_predictions.csv', 'r') as file:
    lines = file.readlines()
    for line in lines:
        values = line.strip().split(',')
        values = values[0:len(values)-1]
        kp_data.extend([int(value) for value in values])
    kp_data.append(kp_data[len(kp_data) - 1])

start_date = f"{YEAR}-01-01-00"
temp = datetime.strptime(start_date, "%Y-%m-%d-%H")

output = {}
for i in range(len(kp_data)):
    date = (temp + timedelta(hours=3))
    date_str = date.strftime("%Y-%m-%d-%H") if i != 0 else start_date
    temp = date 
    output.update({date_str: kp_data[i]})

with open(f'{YEAR}_date_kp.json', 'w') as f:
    json.dump(output, f, indent=4)



