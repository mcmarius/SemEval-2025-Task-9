import csv
from utils import read_file
rws = read_file('parsed_data_f23_h16.csv')
rws2 = []
for r in rws:
    r2 = r
    r2['extracted_hazard'] = ', '.join(r2['extracted_hazard'].split(', ')[0:1])
    r2['extracted_food'] = ', '.join(r2['extracted_food'].split(', ')[0:1])
    rws2.append(r2)
out_file = 'parsed_data_f23_h16_first1.csv'
with open(out_file, 'w', newline='') as csvfile:
     writer = csv.writer(csvfile, delimiter=',', quotechar='"')
     writer.writerow(list(rws2[0].keys()))
     for row in rws2:
         writer.writerow(list(row.values()))
