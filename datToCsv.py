import csv

dat_file = "file.dat"
csv_file = "file.csv"

with open(dat_file, "r") as input_file, open(csv_file, "w", newline='') as output_file:
    writer = csv.writer(output_file, delimiter='|')
    writer.writerow(["uid", "timestamp", "content"])
    for line in input_file:
        parts = line.strip().split(" :: ")
        timestamp = parts[0]
        uid_content = parts[1].split(": ")
        uid = uid_content[0]
        content = uid_content[1]
        writer.writerow([uid, timestamp, content])