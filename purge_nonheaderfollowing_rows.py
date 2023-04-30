import pandas as pd
import csv

# with open("./datasets/sents_merged_cleaned.csv", "r", newline="", encoding="utf-8") as f, open("./datasets/sents_merged_cleaned_pipe.csv", "w", newline="", encoding="utf-8") as o:
# 	reader = csv.reader(f, delimiter=",")
# 	writer = csv.writer(o, delimiter="|")
# 	for row in reader:
# 		writer.writerow(row)

df = pd.read_csv("./datasets/sents_merged_cleaned_pipe.csv", sep="|", header=0, on_bad_lines="skip")

df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

print(df.head())

cols = ["uid", "timestamp", "content", "sentiment"]

df = df[df.apply(lambda x: len(x) == len(cols) and all(x.index == cols), axis=1)]

df.to_csv("./datasets/sents_merged_cleaned_purged.csv", sep="|", index=True)