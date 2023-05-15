import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(dataType):
	if dataType not in ["train", "val", "test"]:
		raise ValueError("dataType must be one of train, val, or test")

	data = pd.read_csv(f"datasets/{dataType}.csv", sep="|")
	data.columns = ["uid", "timestamp", "content", "sentiment"]

	if "neu" in data.columns:
		data = data.rename(columns={"neu": "sentiment"})

	if "nan" in data.columns:
		print(f"nan at row {data[data['nan'].notnull()].index[0]}")
		raise ValueError("nan column found")

	data = data[data["sentiment"].isin(["pos", "neg", "neu"])]

	print(f"n of rows with pos sentiment: {len(data[data['sentiment'] == 'pos'])}")

	return data

def split_data(data, trainRatio, valRatio):
	data = pd.read_csv(data, sep="|")
	data = data.loc[:, ~data.columns.str.startswith("Unnamed")]

	dropped = 0

	# data.columns = ["uid", "timestamp", "content", "sentiment"]
	data.columns = ["uid", "content", "sentiment"]

	try:
		data = data.drop(columns=["nan"])
	except KeyError:
		pass

	data = data.dropna(subset=["content", "sentiment"])
	dropped += len(data) - len(data.dropna(subset=["content", "sentiment"]))

	print(f"Columns: {data.columns}")

	testRatio = 1 - (trainRatio + valRatio)

	trainData, tempData = train_test_split(data, train_size=trainRatio, random_state=69)

	adjValRatio = valRatio / testRatio

	valData, testData = train_test_split(tempData, train_size=adjValRatio, random_state=69)

	trainData.to_csv("datasets/train.csv", index=False, sep="|")
	valData.to_csv("datasets/val.csv", index=False, sep="|")
	testData.to_csv("datasets/test.csv", index=False, sep="|")

	print(f"Train columns: {trainData.columns}")

	print(f"Dropped {dropped} rows")

if __name__ == "__main__":
	#split_data("./datasets/sentAnal/sents_merged_cleaned.csv", 0.75, 0.1)
	split_data("./datasets/all_noTimestamps.csv", 0.75, 0.1)