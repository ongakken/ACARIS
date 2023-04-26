import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(dataType):
	"""
	This function loads data from a CSV file based on the specified data type.
	
	@param dataType The dataType parameter is a string that specifies which dataset to load. It can be
	one of three values: "train", "val", or "test".
	
	@return a pandas DataFrame containing the data from the specified dataType file (train.csv, val.csv,
	or test.csv).
	"""
	if dataType not in ["train", "val", "test"]:
		raise ValueError("dataType must be one of train, val, or test")

	data = pd.read_csv(f"datasets/{dataType}.csv")

	return data

def split_data(data, trainRatio, valRatio):
	data = pd.read_csv(data)

	testRatio = 1 - (trainRatio + valRatio)

	trainData, tempData = train_test_split(data, train_size=trainRatio, random_state=69)

	adjValRatio = valRatio / (valRatio + testRatio)

	valData, testData = train_test_split(tempData, train_size=adjValRatio, random_state=69)

	trainData.to_csv("datasets/train.csv", index=False)
	valData.to_csv("datasets/val.csv", index=False)
	testData.to_csv("datasets/test.csv", index=False)

if __name__ == "__main__":
	split_data("datasets/sents_merged.csv", 0.75, 0.1)