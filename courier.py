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


'''colormaps (for javascript deployment)'''
# south_america = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#69B34C',
#     '3': '#69B34C', 
#     '4': '#ACB334', 
#     '5': '#ACB334', 
#     '6': '#ACB334', 
#     '7': '#FAB733', 
#     '8': '#FAB733', 
#     '9': '#FF8E15', 
#     '10':'#FF4E11'  
# }

# oceania = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#69B34C',
#     '3': '#69B34C', 
#     '4': '#ACB334', 
#     '5': '#ACB334', 
#     '6': '#ACB334', 
#     '7': '#FAB733', 
#     '8': '#FAB733', 
#     '9': '#FF8E15', 
#     '10':'#FF4E11'   
# }

# africa = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#69B34C',
#     '3': '#ACB334', 
#     '4': '#ACB334', 
#     '5': '#ACB334', 
#     '6': '#FAB733', 
#     '7': '#FAB733', 
#     '8': '#FAB733', 
#     '9': '#FF8E15', 
#     '10':'#FF4E11'   
# }

# north_america = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#69B34C',
#     '3': '#ACB334', 
#     '4': '#FAB733', 
#     '5': '#FAB733', 
#     '6': '#FF8E15', 
#     '7': '#FF4E11', 
#     '8': '#FF4E11', 
#     '9': '#FF0D0D', 
#     '10':'#FF0D0D'
# }

# europe = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#ACB334',
#     '3': '#FAB733', 
#     '4': '#FAB733', 
#     '5': '#FF8E15', 
#     '6': '#FF4E11', 
#     '7': '#FF4E11', 
#     '8': '#FF0D0D', 
#     '9': '#FF0D0D', 
#     '10':'#FF0D0D'
# }

# asia = {
#     # k value : color 
#     '0': '#69B34C',
#     '1': '#69B34C',
#     '2': '#ACB334',
#     '3': '#FAB733', 
#     '4': '#FAB733', 
#     '5': '#FF8E15', 
#     '6': '#FF4E11', 
#     '7': '#FF4E11', 
#     '8': '#FF0D0D', 
#     '9': '#FF0D0D', 
#     '10':'#FF0D0D'
# }