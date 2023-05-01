def loadRaw():
	with open("./datasets/rawMsg/msgLog.csv", "r") as f:
		data = [line.strip() for line in f.readlines()]
	return data

def cleanRow(row):
	parts = row.split(",")
	uid = parts[0]
	timestamp = parts[1]
	content = ",".join(parts[2:])
	return [uid, timestamp, content]

def cleanData(rawData):
	cleanedData = [cleanRow(row) for row in rawData]
	return cleanedData

def writeCleaned(cleanedData):
	with open("./datasets/cleanedMsg/msgLog.csv", "w", newline="") as f:
		for row in cleanedData:
			f.write(",".join(row) + "\n")


def main():
	rawData = loadRaw()
	cleanedData = cleanData(rawData)
	writeCleaned(cleanedData)

if __name__ == "__main__":
	main()