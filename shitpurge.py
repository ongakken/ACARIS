import csv

def drop_this_fucking_shit(input, output, string):
	with open(input, "r") as input:
		reader = csv.reader(input, delimiter="|")

		with open(output, "w", newline="") as output:
			writer = csv.writer(output, delimiter="|")

			for row in reader:
				if string not in row[0]:
					writer.writerow(row)


if __name__ == "__main__":
	drop_this_fucking_shit("./datasets/sentAnal/sents_merged_cleaned.csv", "./datasets/sentAnal/sents_merged_cleaned_shitpurged.csv", "Eliza")