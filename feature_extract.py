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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from preprocess import Preprocessor
import pickle
import csv
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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



class FeatExtractor:
    def __init__(self, mdl, nTopics = 10):
        self.sentPipeline = pipeline("sentiment-analysis", model=mdl)
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words="english")
        self.lda = LatentDirichletAllocation(n_components=nTopics, random_state=69)

    def extract_feats(self, msgs):
        feats = {
            "meanWordcount": self.mean_wordcount(msgs),
            "vocabRichness": self.vocab_richness(msgs),
            "meanEmojiCount": self.mean_emoji_count(msgs),
            "meanEmoticonCount": self.mean_emoticon_count(msgs),
            "meanPunctuationCount": self.mean_punctuation_count(msgs),
            "meanSentimentScore": self.mean_sentiment_score(msgs),
            "dominantTopics": self.dominant_topics(msgs),
            "meanResponseTime": self.mean_response_time(msgs),
            "meanMsgCountPerDay": self.mean_msg_count_per_day(msgs),
            "meanLinksPerMsg": self.mean_links_per_msg(msgs),
            "meanCodeSnippetsPerMsg": self.mean_code_snippets_per_msg(msgs),
            "dominantWritingStyles": self.dominant_writing_styles(msgs),
            "meanAbbreviationsPerMsg": self.mean_abbreviations_per_msg(msgs),
            "meanHashtagCountPerMsg": self.mean_hashtag_count_per_msg(msgs),
            "meanAttachmentCountPerMsg": self.mean_attachment_count_per_msg(msgs),
            "meanQuoteCountPerMsg": self.mean_quote_count_per_msg(msgs),
            "dominantActiveHours": self.dominant_active_hours(msgs)
        }
        return feats

    def save_feats_for_later(self, feats, path):
        with open(path, "wb") as f:
            pickle.dump(feats, f)

    def load_feats(self, path):
        with open(path, "rb") as f:
            feats = pickle.load(f)
        return feats

    def read_msgs_from_file(self, path):
        with open(path, "r") as f:
            fieldNames = ["uid", "timestamp", "content", "sentiment"]
            reader = csv.DictReader(f, fieldnames=fieldNames)
            next(reader)
            header = reader.fieldnames
            print(header)
            if header != fieldNames:
                raise ValueError("Invalid header")
            msgs = [row for row in reader if any(row.values())]
            print(msgs)
            for msg in msgs:
                msg.pop(None, None)
        return msgs

    def txt_to_csv(self, txtPath, csvPath, soughtUser):
        pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}):: .+\/(.+?): (.+)"
        with open(txtPath, "r") as infile, open(csvPath, "w") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["timestamp", "uid", "content"])
            writer.writeheader()

            for line in infile:
                match = re.match(pattern, line)
                if match:
                    timestamp, discordTag, content = match.groups()
                    #if "The Butler#5636" not in discordTag and "MarkTheBot#5636" not in discordTag:
                    if soughtUser in discordTag:
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
        if not "uid" in msgs[0] or not "content" in msgs[0] or not "timestamp" in msgs[0]:
            raise ValueError("Fields 'uid' and 'content' must be present in the input data!")

    def remove_none(self, msgs):
        cleaned = []
        for row in msgs:
            if None not in row.values():
                cleaned.append(row)
        msgs = cleaned
        return msgs

    def extract_and_store_feats(self, msgs, path):
        requiredFields = self.get_required_fields()
        self.check_if_fields_exist(msgs, requiredFields)
        feats = self.extract_feats(msgs)
        self.saveFeatsForLater(feats, path)

    ### Feature extraction methods ###

    def mean_wordcount(self, msgs):
        print("Extracting mean wordcount...")
        return np.mean([len(msg["content"].split()) for msg in msgs])
    mean_wordcount.requiredFields = ["content"]

    def vocab_richness(self, msgs):
        print("Extracting vocabulary richness...")
        words = [word for msg in msgs for word in msg["content"].split()]
        uniqueWords = len(set(words))
        return uniqueWords / len(words)
    vocab_richness.requiredFields = ["content"]

    def mean_emoji_count(self, msgs):
        print("Extracting mean emoji count...")
        emojiCount = [len(emojiPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(emojiCount)
    mean_emoji_count.requiredFields = ["content"]

    def mean_emoticon_count(self, msgs):
        print("Extracting mean emoticon count...")
        emoticonCount = [len(emoticonPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(emoticonCount)
    mean_emoticon_count.requiredFields = ["content"]

    def mean_punctuation_count(self, msgs):
        print("Extracting mean punctuation count...")
        punctCount = [sum(char in string.punctuation for char in msg["content"]) for msg in msgs]
        return np.mean(punctCount)
    mean_punctuation_count.requiredFields = ["content"]

    def mean_sentiment_score(self, msgs):
        print("Extracting mean sentiment score...")
        sentMapping = {"pos": 1, "neg": -1, "neu": 0}
        msgs = self.remove_none(msgs)
        if not msgs:
            raise ValueError("Can't proceed without messages!")
        if len(msgs[0]["content"].split()) < 1:
            raise ValueError("Can't proceed without words!")
        sentimentScores = []
        for row in msgs:
            try:
                sentimentScore = sentMapping.get(row.get("sentiment"), 0) if len(row["content"].split()) >= 1 else 0
                sentimentScores.append(sentimentScore)
            except KeyError as e:
                print(f"KeyError: at {row}")
                raise e
        return np.mean(sentimentScores)
    mean_sentiment_score.requiredFields = ["sentiment"]

    def dominant_topics(self, msgs):
        print("Extracting dominant topics...")
        msgTexts = [msg["content"] for msg in msgs]
        docTermMatrix = self.vectorizer.fit_transform(msgTexts)
        self.lda.fit(docTermMatrix)
        topicDistr = np.mean(self.lda.transform(docTermMatrix), axis=0)
        return topicDistr.argmax()
    dominant_topics.requiredFields = ["content"]

    def mean_response_time(self, msgs):
        print("Extracting mean response time...")
        timestamps = [datetime.fromisoformat(msg["timestamp"]) for msg in msgs]
        responseTimes = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        responseTimes = [rt for rt in responseTimes if rt < timedelta(hours=1)]
        return np.mean(responseTimes).total_seconds()
    mean_response_time.requiredFields = ["timestamp"]

    def mean_msg_count_per_day(self, msgs):
        print("Extracting mean message count per day...")
        timestamps = [datetime.fromisoformat(msg["timestamp"]).date() for msg in msgs]
        days = (timestamps[-1] - timestamps[0]).days
        return len(timestamps) / days
    mean_msg_count_per_day.requiredFields = ["timestamp"]

    def mean_links_per_msg(self, msgs):
        print("Extracting mean links per message...")
        linkCount = [len(linkPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(linkCount)
    mean_links_per_msg.requiredFields = ["content"]

    def mean_code_snippets_per_msg(self, msgs):
        print("Extracting mean code snippets per message...")
        codeSnippetCount = [len(codeSnippetPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(codeSnippetCount)
    mean_code_snippets_per_msg.requiredFields = ["content"]

    def dominant_writing_styles(self, msgs):
        print("Extracting dominant writing styles...")
        raise NotImplementedError
    dominant_writing_styles.requiredFields = ["content"]

    def mean_abbreviations_per_msg(self, msgs):
        print("Extracting mean abbreviations per message...")
        abbreviationsCount = [len(abbreviationsPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(abbreviationsCount)
    mean_abbreviations_per_msg.requiredFields = ["content"] 

    def mean_hashtag_count_per_msg(self, msgs):
        print("Extracting mean hashtag count per message...")
        hashtagCount = [len(hashtagPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(hashtagCount)
    mean_hashtag_count_per_msg.requiredFields = ["content"]

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

    # if os.path.splitext("./datasets/msgs.txt")[1] == ".txt":
    #     txt_to_csv("./datasets/msgs.txt", "./datasets/msgs.csv", args.uid)

    msgs = extractor.read_msgs_from_file("./datasets/sents_merged.csv")
    print(msgs[0].keys())
    #msgs = extractor.txt_to_csv("./datasets/msgLog.txt", "./datasets/reknezLog.csv", "Reknez#9257")

    #breakpoint()

    extractor.extract_and_store_feats(msgs, "./pickles/feats.pkl")
