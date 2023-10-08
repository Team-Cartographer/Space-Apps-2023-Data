date_list = {}
kp_val = {}
rows = []
with open('K_p_DATA.dat', mode='r') as file:
    for line in file:
        fields = line.strip().split()
        #print(fields[3])
        rows.append(fields)

#        date_list = fields[0]
#        time_list = fields[1]
#        kp_val = fields[3]
#        print(date_list)
#        print(kp_val)

    file.close()

di = {}

# print(rows)

for row in rows:
    di.update({str(row[0]+ " " +row[1]): int(row[3][0:1])})

date_list = []
time_list = []
# print(di)
for key in di.keys():
    date_time_parts = key.split(" ")

    if len(date_time_parts) >= 2:
        date_list.append(date_time_parts[0])
        time_list.append(date_time_parts[1])

# print(len(date_list))
# print(len(time_list))

kp_list = []
kp_val = list(di.values())
# print(len(kp_val))



