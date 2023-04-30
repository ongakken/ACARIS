import re
import pandas as pd


discordCustomEmojiPattern = re.compile(r"<a?:\w+:\d+>")

def remove_custom_dsc_emoji(text):
	return discordCustomEmojiPattern.sub("", text)

def clean_csv(inputPath, outputPath):
	data = pd.read_csv(inputPath)
	#data = data.loc[:, ~data.columns.str.startswith("Unnamed")]
	data["content"] = data["content"].apply(remove_custom_dsc_emoji)
	data.to_csv(outputPath, index=False)

if __name__ == "__main__":
	clean_csv("./datasets/sents_merged.csv", "./datasets/sents_merged_cleaned.csv")