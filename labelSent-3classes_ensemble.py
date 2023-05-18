from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd
from tqdm import tqdm
import torch
from collections import Counter
import os

maxLen = 512
ckptFreq = 1000

sourcePath = "./datasets/currentlyWorkingDataset/all_noTimestamps.csv"
targetPath = "./datasets/currentlyWorkingDataset/all_noTimestamps_sent_ENSEMBLE.csv"

source = pd.read_csv(sourcePath, sep="|", header=0)
source["content"] = source["content"].astype(str)

if os.path.exists(targetPath):
	target = pd.read_csv(targetPath, sep="|", header=0)

	assert all(target.iloc[:, :-1].eq(source.iloc[:target.shape[0], :-1]).all(axis=1)), "target and source datasets do not match"

	index = target.shape[0]
else:
	target = source.copy()
	target["sentiment"] = ""
	target.to_csc(targetPath, sep="|", index=False)

vader = SentimentIntensityAnalyzer()
flair = TextClassifier.load("en-sentiment")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
twitterRoberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_sentiment(text):
	# sanity conversion
	text = str(text) # just in case

	chunks = [text[i:i+maxLen] for i in range(0, len(text), maxLen)]
	allSents = []

	for chunk in chunks:

		# DistilBERT
		inputs = tokenizer(chunk, return_tensors="pt")
		with torch.no_grad():
			logits = model(**inputs).logits
		probabilities = torch.nn.functional.softmax(logits, dim=-1)
		prediction = probabilities.argmax().item()
		distilbertSent = model.config.id2label[prediction].lower()
		if probabilities.max() < 0.5:
			distilbertSent = "neu"

		# Twitter RoBERTa
		twitterRobertaSent = twitterRoberta(chunk)[0]["label"].lower()

		# TextBlob
		blob = TextBlob(chunk)
		textblobSent = blob.sentiment.polarity
		if textblobSent > 0:
			textblobSent = "pos"
		elif textblobSent < 0:
			textblobSent = "neg"
		else:
			textblobSent = "neu"

		# VADER
		vaderSent = vader.polarity_scores(chunk)["compound"]
		if vaderSent > 0.05:
			vaderSent = "pos"
		elif vaderSent < -0.05:
			vaderSent = "neg"
		else:
			vaderSent = "neu"

		# flair
		sentence = Sentence(chunk)
		flair.predict(sentence)
		flairSent = sentence.labels[0].value.lower()

		sents = [distilbertSent, twitterRobertaSent, textblobSent, vaderSent, flairSent]
		allSents.extend(sents)

	# ensemble vote
	return max(set(allSents), key=allSents.count)

for i in tqdm(range(index, source.shape[0])):
	target.at[i, "sentiment"] = get_sentiment(source.loc[i, "content"])
	if i > 0 and i % ckptFreq == 0:
		target.to_csv(targetPath, sep="|", index=False)

target.to_csv(targetPath, sep="|", index=False)
