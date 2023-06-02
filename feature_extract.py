"""
This module extracts features from tokenized input sequences.
"""

import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from preprocess import Preprocessor
import pickle
import csv
import argparse
import os
from nltk import pos_tag


emojiPattern = re.compile("["
	u"\U0001F600-\U0001F64F" # facial emojis
	u"\U0001F300-\U0001F5FF" # symbol emojis
	u"\U0001F680-\U0001F6FF" # transport and map emojis
	u"\U0001F1E0-\U0001F1FF" # flags (possibly iOS-only)
"]+", flags=re.UNICODE)

emoticonPattern = re.compile(
	r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)"
)

linkPattern = re.compile(
	r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

codeSnippetPattern = re.compile(r"```[\s\S]*?```")

hashtagPattern = re.compile(r"#[^\s]+")

abbreviationsPattern = re.compile(
	r"\b(?:[a-z]*[A-Z][a-z]*(?:[A-Z][a-z]*)?|[A-Z]+(?![a-z]))\b"
)

customDscEmojiPattern = re.compile(r"<(a)?:\w+:\d+>")



class FeatExtractor:
	def __init__(self, mdl, nTopics = 10):
		self.sentPipeline = pipeline("sentiment-analysis", model=mdl)
		self.vectorizer = TfidfVectorizer(stop_words="english")
		self.lda = LatentDirichletAllocation(n_components=nTopics, random_state=69)

	def extract_feats(self, msgs, userID=None):
		if not msgs:
			return []
		if not msgs:
			return []

		msgs = self.removeDscEmoji(msgs)

		feats = {
			#0 current userEmb dimensionality: 12
			"meanWordcount": self.mean_wordcount(msgs),
			"vocabRichness": self.vocab_richness(msgs),
			"meanEmojiCount": self.mean_emoji_count(msgs),
			"meanEmoticonCount": self.mean_emoticon_count(msgs),
			"meanPunctuationCount": self.mean_punctuation_count(msgs),
			"meanSentimentScore": self.mean_sentiment_score(msgs),
			#"dominantTopics": self.dominant_topics(msgs),
			#"meanResponseTime": self.mean_response_time(msgs),
			#"meanMsgCountPerDay": self.mean_msg_count_per_day(msgs),
			"meanLinksPerMsg": self.mean_links_per_msg(msgs),
			#"meanCodeSnippetsPerMsg": self.mean_code_snippets_per_msg(msgs),
			#"dominantWritingStyles": self.dominant_writing_styles(msgs),
			"meanAbbreviationsPerMsg": self.mean_abbreviations_per_msg(msgs),
			#"meanHashtagCountPerMsg": self.mean_hashtag_count_per_msg(msgs),
			"meanSentenceLen": self.mean_sentence_len(msgs),
			"meanNounFreq": self.mean_noun_freq(msgs),
			"meanVerbFreq": self.mean_verb_freq(msgs),
			"meanAdjFreq": self.mean_adjective_freq(msgs)
			#"meanAttachmentCountPerMsg": self.mean_attachment_count_per_msg(msgs),
			#"meanQuoteCountPerMsg": self.mean_quote_count_per_msg(msgs),
			#"dominantActiveHours": self.dominant_active_hours(msgs)
		}

		for k, v in feats.items():
			if v is None:
				print(f"Feature {k} is None for user {userID}")
				raise ValueError
		return feats if all(v is not None for v in feats.values()) else []

	def save_feats_for_later(self, feats, userID, path):
		with open(path, "wb") as f:
			pickle.dump(feats, f)
		
		with open(path + ".csv", "a") as f:
			writer = csv.writer(f)
			writer.writerow([userID] + feats)

	def load_feats(self, path):
		with open(path, "rb") as f:
			feats = pickle.load(f)
		return feats

	def read_msgs_from_file(self, path):
		with open(path, "r", encoding="utf8") as f:
			#fieldNames = ["uid", "timestamp", "content", "sentiment"]
			fieldNames = ["uid", "content", "sentiment"]
			reader = csv.DictReader(f, fieldnames=fieldNames, delimiter="|")
			next(reader)
			header = reader.fieldnames
			print(header)
			if header != fieldNames:
				raise ValueError("Invalid header")
			msgs = [row for row in reader if any(row.values())]
			for msg in msgs:
				msg.pop(None, None)
		# return format: [{"uid": "user#1234", "content": "hello world", "sentiment": "pos"}, ...]
		return msgs

	def group_msgs_by_user(self, msgs):
		groups = {}
		for msg in msgs:
			uid = msg["uid"]
			content = msg["content"]
			if uid not in groups:
				groups[uid] = []
			groups[uid].append(content)
		return groups

	def txt_to_csv(self, txtPath, csvPath, soughtUser):
		pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}):: .+\/(.+?): (.+)"
		alsoPattern = r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s::\s([\w#]+):\s(.+)$"
		with open(txtPath, "r") as infile, open(csvPath, "w") as outfile:
			writer = csv.DictWriter(outfile, fieldnames=["uid", "timestamp", "content"], delimiter="|")
			writer.writeheader()

			for line in infile:
				match = re.match(pattern, line)
				alsoMatch = re.match(alsoPattern, line)
				if match:
					timestamp, discordTag, content = match.groups()
					if "The Butler#5636" not in discordTag and "MarkTheBot#5636" not in discordTag:
						writer.writerow({"uid": discordTag, "timestamp": timestamp, "content": content})
				elif alsoMatch:
					timestamp, discordTag, content = alsoMatch.groups()
					if "The Butler#5636" not in discordTag and "MarkTheBot#5636" not in discordTag:
						writer.writerow({"uid": discordTag, "timestamp": timestamp, "content": content})
		print(f"Saved {csvPath}")

	def cleanup_csv(self, csvPath):
		raise NotImplementedError

	def get_required_fields(self):
		featMethods = [
			self.mean_wordcount,
			self.vocab_richness,
			self.mean_emoji_count,
			self.mean_emoticon_count,
			self.mean_punctuation_count,
			self.mean_sentiment_score,
			self.dominant_topics,
			self.mean_response_time,
			self.mean_msg_count_per_day,
			self.mean_links_per_msg,
			self.mean_code_snippets_per_msg,
			self.dominant_writing_styles,
			self.mean_abbreviations_per_msg,
			self.mean_hashtag_count_per_msg,
			self.mean_attachment_count_per_msg,
			self.mean_quote_count_per_msg,
			self.dominant_active_hours
		]

		requiredFields = set()
		for method in featMethods:
			fields = getattr(method, "requiredFields", [])
			requiredFields.update(fields)

		return list(requiredFields)

	def check_if_fields_exist(self, msgs, requiredFields):
		#if not "uid" in msgs[0] or not "content" in msgs[0] or not "timestamp" in msgs[0]:
		print(f"msgs[0]: {msgs[0]}")
		print(f"msgs: {msgs}")
		if not "uid" in msgs[0] or not "content" in msgs[0]:
			#raise ValueError("Fields 'uid', 'content', and 'timestamp' must be present in the input data!")
			raise ValueError("Fields 'uid' and 'content' must be present in the input data!")

	def remove_none(self, msgs):
		cleaned = []
		for row in msgs:
			if None not in row.values():
				cleaned.append(row)
		msgs = cleaned
		return msgs

	def removeDscEmoji(self, msgs):
		clnMsgs = []
		for msg in msgs:
			cln = customDscEmojiPattern.sub("", msg["content"])
			clnMsg = msg.copy()
			clnMsg["content"] = cln
			clnMsgs.append(clnMsg)
			msgs = clnMsgs
		return msgs

	def extract_and_store_feats(self, msgs, userID=None, path="./pickles/"):
		requiredFields = self.get_required_fields()
		#self.check_if_fields_exist(msgs, requiredFields) # disabled due to restructuring of msgs
		groupedMsgs = self.group_msgs_by_user(msgs)

		if userID is not None:
			#if userID not in groupedMsgs or len(groupedMsgs[userID]) < 25:
			if len(msgs) < 25:
				return None
			#msgs = groupedMsgs[userID] # disabled for now
			assert all(msg["uid"] == userID for msg in msgs), f"msgs contains messages from multiple users: {[msg['uid'] for msg in msgs]}" #sanity check to ensure that all dicts in msgs belong to the same user
			feats = self.extract_feats(msgs=msgs, userID=userID)
			if not feats:
				print(f"{userID} has 0 feats!")
				return None
			else:
				print(f"Extracted {len(feats)} features from {userID}")
				return feats

		else:
			feats = {}
			for uid, userMsgs in groupedMsgs.items():
				if len(userMsgs) < 25:
					print(f"Skipping {uid} because they have less than 25 messages")
					continue
				feats[uid] = self.extract_feats(msgs=userMsgs, userID=uid)
				print(f"Extracted {len(feats)} features from {uid}")

				picklePath = os.path.join(path, f"{uid.split('#')[0]}_feats.pkl") # using only the first part of the uid as the filename
				self.save_feats_for_later(feats[uid], userID, picklePath)
			return feats

	### Feature extraction methods ###

	def mean_wordcount(self, msgs):
		print("Extracting mean wordcount...")
		return round(np.mean([len(msg["content"].split()) for msg in msgs]), 2)
	mean_wordcount.requiredFields = ["content"]

	def vocab_richness(self, msgs):
		print("Extracting vocabulary richness...")
		words = [word for msg in msgs for word in msg["content"].split()]
		uniqueWords = len(set(words))
		if len(words) == 0:
			return 0
		else:
			return round(uniqueWords / len(words), 2)
	vocab_richness.requiredFields = ["content"]

	def mean_emoji_count(self, msgs):
		print("Extracting mean emoji count...")
		emojiCount = [len(emojiPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(emojiCount), 2)
	mean_emoji_count.requiredFields = ["content"]

	def mean_emoticon_count(self, msgs):
		print("Extracting mean emoticon count...")
		emoticonCount = [len(emoticonPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(emoticonCount), 2)
	mean_emoticon_count.requiredFields = ["content"]

	def mean_punctuation_count(self, msgs):
		print("Extracting mean punctuation count...")
		punctCount = [sum(char in string.punctuation for char in msg["content"]) for msg in msgs]
		return round(np.mean(punctCount), 2)
	mean_punctuation_count.requiredFields = ["content"]

	def mean_sentiment_score(self, msgs):
		print("Extracting mean sentiment score...")
		sentMapping = {"pos": 2, "neg": 0, "neu": 1} # adjusted due to CrossEntropyLoss
		msgs = self.remove_none(msgs)
		if not msgs or len(msgs[0]["content"].split()) < 1:
			print(f"Empty msgs!")
			return None
		sentimentScores = []
		for row in msgs:
			try:
				sentimentScore = sentMapping.get(row.get("sentiment"), 0) if len(row["content"].split()) >= 1 else 0
				sentimentScores.append(sentimentScore)
			except KeyError as e:
				print(f"KeyError: at {row}")
				raise e
		return round(np.mean(sentimentScores), 2)
	mean_sentiment_score.requiredFields = ["sentiment"]

	def dominant_topics(self, msgs):
		print("Extracting dominant topics...")
		msgTexts = [msg["content"] for msg in msgs]
		docTermMatrix = self.vectorizer.fit_transform(msgTexts)
		self.lda.fit(docTermMatrix)
		topicDistr = np.mean(self.lda.transform(docTermMatrix), axis=0)
		return int(topicDistr.argmax())
	dominant_topics.requiredFields = ["content"]

	def mean_response_time(self, msgs):
		print("Extracting mean response time...")
		timestamps = [datetime.fromisoformat(msg["timestamp"]) for msg in msgs]
		responseTimes = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
		responseTimes = [rt for rt in responseTimes if rt < timedelta(hours=1)]
		if not responseTimes:
			return 0.0
		return round(np.mean(responseTimes).total_seconds(), 2)
	mean_response_time.requiredFields = ["timestamp"]

	def mean_msg_count_per_day(self, msgs):
		print("Extracting mean message count per day...")
		timestamps = [datetime.fromisoformat(msg["timestamp"]).date() for msg in msgs]
		if len(timestamps) < 2 or timestamps[-1] == timestamps[0] or not timestamps:
			return 0.0
		days = (timestamps[-1] - timestamps[0]).days
		return round(len(timestamps) / days, 2)
	mean_msg_count_per_day.requiredFields = ["timestamp"]

	def mean_links_per_msg(self, msgs):
		print("Extracting mean links per message...")
		linkCount = [len(linkPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(linkCount), 2)
	mean_links_per_msg.requiredFields = ["content"]

	def mean_code_snippets_per_msg(self, msgs):
		print("Extracting mean code snippets per message...")
		codeSnippetCount = [len(codeSnippetPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(codeSnippetCount), 2)
	mean_code_snippets_per_msg.requiredFields = ["content"]

	def dominant_writing_styles(self, msgs):
		print("Extracting dominant writing styles...")
		raise NotImplementedError
	dominant_writing_styles.requiredFields = ["content"]

	def mean_abbreviations_per_msg(self, msgs):
		print("Extracting mean abbreviations per message...")
		abbreviationsCount = [len(abbreviationsPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(abbreviationsCount), 2)
	mean_abbreviations_per_msg.requiredFields = ["content"]

	def mean_hashtag_count_per_msg(self, msgs):
		print("Extracting mean hashtag count per message...")
		hashtagCount = [len(hashtagPattern.findall(msg["content"])) for msg in msgs]
		return round(np.mean(hashtagCount), 2)
	mean_hashtag_count_per_msg.requiredFields = ["content"]

	def mean_sentence_len(self, msgs):
		print("Extracting mean sentence length...")
		content = " ".join([msg["content"] for msg in msgs])
		sentences = re.findall(r'\w[^.!?]*[.!?]', content)
		nSentences = len(sentences)
		nWords = len(re.findall(r'\w+', content))
		try:
			return nWords / nSentences
		except ZeroDivisionError:
			return 0
	mean_sentence_len.requiredFields = ["content"]

	def mean_noun_freq(self, msgs):
		print("Extracting mean noun frequency...")
		words = [word for msg in msgs for word in msg["content"].split()]
		tagged = pos_tag(words)
		pos = Counter(tag for _, tag in tagged)
		try:
			return pos["NN"] / sum(pos.values())
		except ZeroDivisionError:
			return 0
	mean_noun_freq.requiredFields = ["content"]

	def mean_verb_freq(self, msgs):
		print("Extracting mean verb frequency...")
		words = [word for msg in msgs for word in msg["content"].split()]
		tagged = pos_tag(words)
		pos = Counter(tag for _, tag in tagged)
		try:
			return pos["VB"] / sum(pos.values())
		except ZeroDivisionError:
			return 0
	mean_verb_freq.requiredFields = ["content"]

	def mean_adjective_freq(self, msgs):
		print("Extracting mean adjective frequency...")
		words = [word for msg in msgs for word in msg["content"].split()]
		tagged = pos_tag(words)
		pos = Counter(tag for _, tag in tagged)
		try:
			return pos["JJ"] / sum(pos.values())
		except ZeroDivisionError:
			return 0
	mean_adjective_freq.requiredFields = ["content"]

	def mean_attachment_count_per_msg(self, msgs):
		print("Extracting mean attachment count per message...")
		attachmentCount = [len(msg["attachments"]) for msg in msgs]
		return np.mean(attachmentCount)
	mean_attachment_count_per_msg.requiredFields = ["attachments"]

	def mean_quote_count_per_msg(self, msgs):
		print("Extracting mean quote count per message...")
		quoteCount = [len(msg["attachments"].split(",")) if msg["attachments"] else 0 for msg in msgs]
		return np.mean(quoteCount)
	mean_quote_count_per_msg.requiredFields = ["attachments"]

	def dominant_active_hours(self, msgs):
		print("Extracting dominant active hours...")
		raise NotImplementedError
	dominant_active_hours.requiredFields = ["timestamp"]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--uid", type=str, required=False, help="UID of the user for all messages")
	args = parser.parse_args()

	mdl = "distilbert-base-uncased"
	extractor = FeatExtractor(mdl)

	if os.path.splitext("./datasets/messagesNew.txt")[1] == ".txt":
		extractor.txt_to_csv("./datasets/messagesNew.txt", "./datasets/msgsNew.csv", args.uid)