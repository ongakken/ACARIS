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
import transformers
import string
from alert import send_alert

maxLen = 510 # 512 - 2 because of special tokens
ckptFreq = 1000

sourcePath = "./datasets/currentlyWorkingDataset/all_noTimestamps.csv"
targetPath = "./datasets/currentlyWorkingDataset/all_noTimestamps_sent_ENSEMBLE.csv"

source = pd.read_csv(sourcePath, sep="|", header=0)
source = source.dropna()
source = source[source["content"].str.strip() != ""].reset_index(drop=True)
source = source[source["content"].str.strip().apply(lambda x: any(c in string.printable for c in x))].reset_index(drop=True)
source["content"] = source["content"].astype(str)

if os.path.exists(targetPath):
	target = pd.read_csv(targetPath, sep="|", header=0, dtype={"sentiment": str})
	target["content"] = target["content"].astype(str)

	if not target.empty:
		sharedCols = list(set(target.columns) & set(source.columns) - {"sentiment"})
		nonmatching_rows = target[sharedCols].ne(source[sharedCols].iloc[:target.shape[0]]).any(axis=1)
		if nonmatching_rows.any():
			print("These rows do not match:")
			print("In source:")
			print(source.iloc[nonmatching_rows.values])
			print("In target:")
			print(target[nonmatching_rows])
		assert all(target[sharedCols].eq(source[sharedCols].iloc[:target.shape[0], :]).all(axis=1)), "target and source datasets do not match"
		index = target.shape[0]
	else:
		index = 0
else:
	target = source.copy()
	target["sentiment"] = ""
	target.to_csv(targetPath, sep="|", index=False)
	index = 0

vader = SentimentIntensityAnalyzer()
flair = TextClassifier.load("en-sentiment")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
twitterRoberta = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest")
twitterRobertaTokenizer = transformers.AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

def get_sentiment(text):
	try:
		# sanity conversion
		text = str(text) # just in case

		chunks = [chunk.strip() for chunk in (text[i:i+maxLen] for i in range(0, len(text), maxLen)) if chunk.strip()]
		chunks = [chunk for chunk in chunks if chunk.strip()] # this is hilarious!
		#print(f"Chunks: {chunks}")
		allSents = []

		for chunk in chunks:
			#print(f"len after tokenization: {len(twitterRobertaTokenizer(chunk)['input_ids'])}")
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
			if len(twitterRobertaTokenizer(chunk)) > twitterRobertaTokenizer.model_max_length - 2:
				print("token", twitterRobertaTokenizer(chunk))
				print("token len:", len(twitterRobertaTokenizer(chunk).input_ids[0]))
				chunk = " ".join(chunk.split()[:twitterRobertaTokenizer.model_max_length - 2])
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
			try:
				flairSent = sentence.labels[0].value.lower()
			except IndexError:
				print(f"IndexError for chunk {chunk}")

			sents = [distilbertSent, twitterRobertaSent, textblobSent, vaderSent, flairSent]
			allSents.extend(sents)

		# ensemble vote
		vote = max(set(allSents), key=allSents.count)
		if vote == "positive":
			return "pos"
		elif vote == "negative":
			return "neg"
		else:
			return "neu"
	except Exception as e:
		print(f"Exception for text {text}: {e}")
		send_alert("Sentiment analysis failed", f"Exception for text {text}: {e}", "critical", 50000)

for i in tqdm(range(index, source.shape[0])):
	target.at[i, "sentiment"] = get_sentiment(source.loc[i, "content"])
	if i > 0 and i % ckptFreq == 0:
		target.to_csv(targetPath, sep="|", index=False)

target.to_csv(targetPath, sep="|", index=False)
send_alert("Sentiment analysis finished")
