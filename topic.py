from bertopic import BERTopic
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer, util
import numpy as np

# load the txt file and make it a list, one sentence per element
text = open("tinyorigins.txt", "r").read()
text = "\n".join(text.split(". "))
text = text.split('\n')
print(len(text)) # print the length of the file

# hyperparams
min_cluster_size = 2
min_samples = 1
nr_topics = 5
n_gram_range = (1, 1)
embedding_model = SentenceTransformer("bert-base-nli-mean-tokens")
UMAP_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=69)
HDBSCAN_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
CountVectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english", max_df=0.9, min_df=0.1)
ClassTfidfTransformer_model = ClassTfidfTransformer()

topic = BERTopic(language="english", calculate_probabilities=True, verbose=True, embedding_model=embedding_model, umap_model=UMAP_model, hdbscan_model=HDBSCAN_model, vectorizer_model=CountVectorizer_model, n_gram_range=n_gram_range, min_topic_size=min_cluster_size, nr_topics=nr_topics, ctfidf_model=ClassTfidfTransformer)
topics, probs = topic.fit_transform(text)

print(topic.get_topic_info())