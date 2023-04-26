import pandas as pd

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