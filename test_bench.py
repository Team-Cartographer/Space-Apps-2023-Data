date_list = {}
kp_val = {}
rows = []
with open('C://Users//zhasi//Downloads//isgi_data_1696720276_541747//Kp_2016-01-01_2021-12-31_D.dat', mode='r') as file:
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

print(rows)

for row in rows:
    di.update({str(row[0]+ " " +row[1]): int(row[3][0:1])})

print(di)

