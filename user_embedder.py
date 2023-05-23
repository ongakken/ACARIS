"""
This mod handles user embeddings.
"""

import torch
from feature_extract import FeatExtractor


class UserEmbedder:
	def __init__(self, userEmbeddingSize=11):
		self.mdl = "distilbert-base-uncased"
		self.userEmbeddings = {}
		self.userEmbeddingSize = userEmbeddingSize
		self.extractor = FeatExtractor(self.mdl)
		self.msgs = self.extractor.read_msgs_from_file("./datasets/currentlyWorkingDataset/all_noTimestamps_sent_ENSEMBLE.csv")

		#0 format of self.msgs is a list of dicts, each dict equal to a row in the csv (i.e. a message) (i.e. a dict of the format {"userID": userID, "content": content, "sentiment": sentiment})
	# BUG: self.msgs is empty, so no user embeddings are created
	def get_user_embedding(self, userID):
		if userID not in self.userEmbeddings:
			print(f"Creating user embedding for user {userID}")
			print(f"self.msgs: {self.msgs}")
			self.msgs = [msg for msg in self.msgs if msg["uid"] == userID]
			self.msgs = [msg for msg in self.msgs if len(msg["content"]) >= 25] 
			print(f"self.msgs: {self.msgs}")
			breakpoint()
			# filter out messages with less than 25 characters
			feats = self.extractor.extract_and_store_feats(self.msgs, userID)
			if feats is not None:
				self.userEmbeddings[userID] = self.create_user_embedding(feats)
			else:
				print(f"Skipping user {userID} due to None feats")
				return None
		return self.userEmbeddings[userID]

	def update_user_embedding(self, userID):
		feats = self.extractor.extract_and_store_feats(self.msgs, userID)
		self.userEmbeddings[userID] = self.create_user_embedding(feats)

	def create_user_embedding(self, feats):
		if not feats:
			return None

		for k, v in feats.items():
			if v is None:
				print(f"None value for {k}")
				raise ValueError
		print(f"\nfeats: {feats} of len {len(feats)}\n")
		feats = [v for v in feats.values() if isinstance(v, (int,float))]
		print(f"\nfeats: {feats} of len {len(feats)}\n")
		emb = torch.tensor(feats, dtype=torch.float)
		return emb