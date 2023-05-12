import csv

emoToSent = {
    "admiration": "pos",
    "amusement": "pos",
    "anger": "neg",
    "annoyance": "neg",
    "approval": "pos",
    "caring": "pos",
    "confusion": "neu",
    "curiosity": "pos",
    "desire": "pos",
    "disappointment": "neg",
    "disapproval": "neg",
    "disgust": "neg",
    "embarrassment": "neg",
    "excitement": "pos",
    "fear": "neg",
    "gratitude": "pos",
    "grief": "neg",
    "joy": "pos",
    "love": "pos",
    "nervousness": "neg",
    "optimism": "pos",
    "pride": "pos",
    "realization": "pos",
    "relief": "pos",
    "remorse": "neg",
    "sadness": "neg",
    "surprise": "pos",
    "neutral": "neu"
}

with open("./datasets/msgsNewLabeled.csv", "r") as f_in, open("./datasets/sentAnal/sents.csv", "w", newline="") as f_out:
	reader = csv.reader(f_in, delimiter="|")
	writer = csv.writer(f_out, delimiter="|")
	for row in reader:
		emoLabel = row[3]
		sentLabel = emoToSent.get(emoLabel, "neu")
		writer.writerow([row[0], row[1], row[2], sentLabel])