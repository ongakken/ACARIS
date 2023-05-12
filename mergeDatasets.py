import pandas as pd

df1 = pd.read_csv("./datasets/sentAnal/sents_merged_cleaned.csv", encoding="utf-8", delimiter="|")
df2 = pd.read_csv("./datasets/sentAnal/sents_merged.csv", encoding="utf-8", delimiter="|")

merged = pd.concat([df1, df2], ignore_index=True)

# only do this if headers match !!! jackass LOL
merged.to_csv("./datasets/sentAnal/sents_merged_cleaned_expanded.csv", index=False, sep="|")