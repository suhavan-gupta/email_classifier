import csv

file_object1 = csv.reader(open("final_test_data.csv"))
file_object2 = csv.reader(open("output.csv"))

output_file_object = csv.writer(open("stats.csv", "w"))

url = []
i = 1
for url, tag in zip(file_object1, file_object2):
    output_file_object.writerow([i, tag, url[4], url[5]])
    i = i + 1
