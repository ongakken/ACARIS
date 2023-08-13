'''
The module for exploratory data analysis of the new datasets.
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from IPython.display import display
import gc
import re
from collections import defaultdict



def basic_stats(ds):
	'''
	Extract basic stats about a dataset
	'''
	
	# basic
	print(f"Distribution of types:\n{ds['type'].value_counts()}")
	print(f"Number of unique types: {len(ds['type'].unique())}")

	# posts
	ds["postLengths"] = ds["posts"].apply(len)
	print(f"Post length describe:\n{ds.groupby('type')['postLengths'].describe()}")

def visualize(ds):
	'''
	Visualize the dataset
	'''
	plt.figure(figsize=(16, 6)) # width, height of plots

	# distribution of types
	sns.countplot(data=ds, x="type", order=ds["type"].value_counts().index, palette="viridis") # order by value counts. "viridis" is a color palette
	plt.title("Distribution of types")
	plt.ylabel("Count")
	plt.xlabel("Type")
	plt.grid(axis="y")
	display(plt.show())

	# histogram for post lengths
	plt.figure(figsize=(16, 6))
	sns.histplot(ds["postLengths"], bins=50, color="skyblue", kde=True) # bins is the number of bars. kde is the density line
	plt.title("Distribution of post lengths")
	plt.ylabel("Count")
	plt.xlabel("Post length")
	plt.grid(axis="y")
	display(plt.show())

	# mean number of words per post
	ds["nWords"] = ds["posts"].apply(lambda x: len(tokenize(x)))
	meanWordsPerPost = ds.groupby("type")["nWords"].mean().sort_values(ascending=False)
	plt.figure(figsize=(16, 6))
	sns.barplot(x=meanWordsPerPost.index, y=meanWordsPerPost.values, palette="viridis")
	plt.title("Mean number of words per post")
	plt.ylabel("Mean number of words")
	plt.xlabel("Type")
	plt.grid(axis="y")
	display(plt.show())

def tokenize(posts):
	'''
	Tokenize the posts
	'''
	return posts.lower().replace("|||", " ").split()

def gen_wordclouds(ds, type, nWords):
	'''
	Generate wordclouds for given type
	'''
	stopWords = set(stopwords.words("english"))

	for t in type:
		words = ds[ds["type"] == t]["posts"].apply(tokenize).explode().tolist()
		words = [str(word) for word in words]
		wordcloud = WordCloud( # instantiate a dark-themed wordcloud that ingores stopwords
			stopwords=stopWords,
			background_color="black",
			width=1600,
			height=800,
			max_words=nWords,
		).generate(" ".join(words))

		plt.figure(figsize=(12, 10))
		plt.imshow(wordcloud, interpolation="bilinear")
		plt.axis("off")
		plt.title(f"Wordcloud for {t} type")
		display(plt.show())

def plot_word_frequency(ds, words):
	'''
	Plot the frequency of words given as input
	'''
	# count the frequency of words for each type
	totalCounts = defaultdict(int)
	typeWordCounts = defaultdict(lambda: defaultdict(int))

	typeCounts = ds["type"].value_counts()

	def count_words(row):
		post = row["posts"]
		type = row["type"]
		for w in words:
			count = str(post).split().count(w)
			totalCounts[w] += count
			typeWordCounts[type][w] += count

	ds.apply(count_words, axis=1)

	meanFreqs = {word: {type: counts.get(word, 0) / typeCounts[type] for type, count in counts.items()} for word, counts in typeWordCounts.items()}

	gc.collect()

	# plot the mean frequency of words for each type and word
	for word, freq in meanFreqs.items():
		plt.figure(figsize=(16, 6))
		sns.barplot(x=list(freq.keys()), y=list(freq.values()), palette="viridis")
		plt.title(f"Mean frequency of \"{word}\" for each type")
		plt.ylabel("Frequency (mean)")
		plt.xlabel("Type")
		plt.grid(axis="y")
		display(plt.show())

	# plot the count of words for each type and word
	for word, count in typeWordCounts.items():
		plt.figure(figsize=(16, 6))
		sns.barplot(x=list(count.keys()), y=list(count.values()), palette="viridis")
		plt.title(f"Count of \"{word}\" for each type")
		plt.ylabel("Count (total)")
		plt.xlabel("Type")
		plt.grid(axis="y")
		display(plt.show())
	

if __name__ == "__main__":
	ds = pd.read_csv("datasets/personality/MBTI/mbti_all_deduplicated_filtered.csv", dtype={"type": "category", "posts": "string"})

	# sanity checks
	print(ds.head())
	print(ds["type"].value_counts())
	print("Shape:", ds.shape)
	print("Columns:", ds.columns)
	print("Describe:", ds.describe())

	basic_stats(ds)
	visualize(ds)
	# gen_wordclouds(ds, ["ISTJ", "ISFJ", "INFJ", "INTJ", "ISTP", "ISFP", "INFP", "INTP", "ESTP", "ESFP", "ENFP", "ENTP", "ESTJ", "ESFJ", "ENFJ", "ENTJ"], 100) # explore all types
	plot_word_frequency(ds, ["sex"])