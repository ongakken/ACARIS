import csv
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis


userTopics = {}

def get_user_msgs(userID, csvPath):
	msgs = []
	with open(csvPath, "r") as f_in:
		reader = csv.reader(f_in, delimiter="|")
		for row in reader:
			if row[0] == userID:
				msgs.append(row[2])
	return msgs

def preprocess_msgs(msgs):
	stopWords = set(stopwords.words("english"))
	numPattern = re.compile(r"\d")
	processedMsgs = []
	for msg in msgs:
		words = re.findall(r"\b(?!\d+\b)\w+\b", msg)
		words = [word.lower() for word in words if word.lower() not in stopWords and "smiling" not in word.lower() and "eyes" not in word.lower() and "face" not in word.lower() and "https" not in word.lower() and "grinning" not in word.lower() and "clyde" not in word.lower() and "unclyde" not in word.lower()]
		processedMsgs.append(words)
	return processedMsgs

def get_topics(nTopics, passes, corpus, dict):
	lda = LdaModel(corpus=corpus, id2word=dict, num_topics=nTopics, passes=passes)
	topics = lda.print_topics(num_words=10)
	return topics

def print_topics(lda, topics):
	for topic in topics:
		topicID = topic[0]
		topicWeight = lda.alpha[topicID]
		print(f"Topic {topicID} (weight: {topicWeight}):")
		keywords = topic[1].split(" + ")
		for keyword in keywords:
			weight, word = keyword.split("*")
			word = word.strip()
			weight = float(weight.strip())
			print(f"\t{word} ({weight})")

def get_user_topics(lda, dict, stopWords, processedMsgs):
	userTopics = {}
	for msg in processedMsgs:
		bow = dict.doc2bow(msg)
		topicDist = lda.get_document_topics(bow)
		topTopics = sorted(topicDist, key=lambda x: x[1], reverse=True)[:3]
		for topic in topTopics:
			topicID = topic[0]
			topicWeight = topic[1]
			if topicID not in userTopics:
				userTopics[topicID] = 0
			userTopics[topicID] += topicWeight
	return userTopics

def get_most_common_sentiment(userID, csvPath):
	sents = []
	with open(csvPath, "r") as f_in:
		reader = csv.reader(f_in, delimiter="|")
		for row in reader:
			if row[0] == userID:
				sents.append(row[3])
	sentCounts = Counter(sents)
	mostCommonSent = sentCounts.most_common(1)[0]
	return mostCommonSent

def generate_wordcloud(processedMsgs):
	processedMsgs = [word for msg in processedMsgs for word in msg]
	wordcloud = WordCloud(width=1920, height=1080, background_color="black").generate_from_frequencies(Counter(processedMsgs))
	plt.figure(figsize=(16,9))
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.show()

def main():
	stopw = set(stopwords.words("english"))
	#msgs = get_user_msgs("Reknez#9257", "./datasets/sentAnal/sents_merged_sents_merged_cleaned_expanded.csv")
	msgs = get_user_msgs("simtoon", "./datasets/sentAnal/sents_merged_cleaned_expanded.csv")
	# msgs = get_user_msgs("Clyde#0000", "./datasets/sentAnal/sents_merged_cleaned_expanded.csv")
	#msgs = get_user_msgs("Reknez#9257", "./datasets/sentAnal/sents_merged_cleaned_expanded.csv")
	processedMsgs = preprocess_msgs(msgs)
	generate_wordcloud(processedMsgs)
	dict = corpora.Dictionary(processedMsgs)
	corpus = [dict.doc2bow(words) for words in processedMsgs]
	lda = LdaModel(corpus=corpus, id2word=dict, num_topics=50, passes=20)
	topics = lda.print_topics(num_words=10)

	vis = gensimvis.prepare(lda, corpus, dict)
	pyLDAvis.save_html(vis, fileobj="lda.html")

	userTopicProbs = get_user_topics(lda, dict, stopw, processedMsgs)
	userTopics["Reknez#9257"] = userTopicProbs

	for userID, topics in userTopics.items():
		sortedTopics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]

		for topic, topicProb in sortedTopics:
			print(f"{userID} is most likely to talk about topic {topic} ({topicProb})")

	print(f"The most common sentiment for Reknez#9257 is {get_most_common_sentiment('Reknez#9257', './datasets/sentAnal/sents_merged_cleaned.csv')}")

if __name__ == "__main__":
	main()