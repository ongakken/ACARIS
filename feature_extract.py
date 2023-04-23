"""
This module extracts features from tokenized input sequences.
"""

import torch
from transformers import AutoModel, AutoTokenizer, pipeline
import re
import string
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from preprocess import Preprocessor

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
            "meanWordcount": self.meanWordcount(msgs),
            "vocabRichness": self.vocabRichness(msgs),
            "meanEmojiCount": self.meanEmojiCount(msgs),
            "meanEmoticonCount": self.meanEmoticonCount(msgs),
            "meanPunctuationCount": self.meanPunctuationCount(msgs),
            "meanSentimentScore": self.meanSentimentScore(msgs),
            "dominantTopics": self.dominantTopics(msgs),
            "meanResponseTime": self.meanResponseTime(msgs),
            "meanMsgCountPerDay": self.meanMsgCountPerDay(msgs),
            "meanLinksPerMsg": self.meanLinksPerMsg(msgs),
            "meanCodeSnippetsPerMsg": self.meanCodeSnippetsPerMsg(msgs),
            "dominantWritingStyles": self.dominantWritingStyles(msgs),
            "meanAbbreviationsPerMsg": self.meanAbbreviationsPerMsg(msgs),
            "meanHashtagCountPerMsg": self.meanHashtagCountPerMsg(msgs),
            "meanAttachmentCountPerMsg": self.meanAttachmentCountPerMsg(msgs),
            "meanQuoteCountPerMsg": self.meanQuoteCountPerMsg(msgs),
            "dominantActiveHours": self.dominantActiveHours(msgs)
        }
        return feats

    def mean_wordcount(self, msgs):
        return np.mean([len(msg) for msg in msgs])

    def vocab_richness(self, msgs):
        words = [word for msg in msgs for word in msg["content"].split()]
        uniqueWords = len(set(words))
        return uniqueWords / len(words)

    def mean_emoji_count(self, msgs):
        emojiCount = [len(emojiPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(emojiCount)

    def mean_emoticon_count(self, msgs):
        emoticonCount = [len(emoticonPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(emoticonCount)

    def mean_punctuation_count(self, msgs):
        punctCount = [sum(char in string.punctuation for char in msg["content"]) for msg in msgs]
        return np.mean(punctCount)

    def mean_sentiment_score(self, msgs):
        sentimentScores = [self.sentPipeline(msg["content"])[0]["score"] for msg in msgs]
        return np.mean(sentimentScores)

    def dominant_topics(self, msgs):
        msgTexts = [msg["content"] for msg in msgs]
        dtm = self.vectorizer.fit_transform(msgTexts)
        self.lda.fit(dtm)
        topicDistr = np.mean(self.lda.transform(dtm), axis=0)
        return topicDistr.argmax()

    def mean_response_time(self, msgs):
        timestamps = [datetime.fromisoformat(msg["timestamp"]) for msg in msgs]
        responseTimes = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        responseTimes = [rt for rt in responseTimes if rt < timedelta(hours=1)]
        return np.mean(responseTimes).total_seconds()

    def mean_msg_count_per_day(self, msgs):
        timestamps = [datetime.fromisoformat(msg["timestamp"]).date() for msg in msgs]
        days = (timestamps[-1] - timestamps[0]).days
        return len(timestamps) / days

    def mean_links_per_msg(self, msgs):
        linkCount = [len(linkPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(linkCount)

    def mean_code_snippets_per_msg(self, msgs):
        codeSnippetCount = [len(codeSnippetPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(codeSnippetCount)

    def dominant_writing_styles(self, msgs):
        raise NotImplementedError

    def mean_abbreviations_per_msg(self, msgs):
        abbreviationsCount = [len(abbreviationsPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(abbreviationsCount)

    def mean_hashtag_count_per_msg(self, msgs):
        hashtagCount = [len(hashtagPattern.findall(msg["content"])) for msg in msgs]
        return np.mean(hashtagCount)

    def mean_attachment_count_per_msg(self, msgs):
        attachmentCount = [len(msg["attachments"]) for msg in msgs]
        return np.mean(attachmentCount)

    def mean_quote_count_per_msg(self, msgs):
        quoteCount = [msg["content"].count(">") for msg in msgs]
        return np.mean(quoteCount)

    def dominant_active_hours(self, msgs):
        raise NotImplementedError

if __name__ == "__main__":
    mdl = "distilbert-base-uncased"
    preprocessor = Preprocessor(mdl)
    extractor = FeatExtractor(mdl)

    tokens = preprocessor.tokenize("Fuck me slowly!")

    feats = extractor.extract_feats(tokens)
    print(f"Feats:\n{feats}")