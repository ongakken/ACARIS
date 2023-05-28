"""
This mod handles user embeddings.
"""

import torch
import os
from tqdm import tqdm
from feature_extract import FeatExtractor

class UserEmbedder:
	def __init__(self, userEmbeddingSize) -> None:
		self.mdl = "distilbert-base-uncased"
		self.userEmbeddings = {}
		self.userEmbeddingSize = userEmbeddingSize
		self.extractor = FeatExtractor(self.mdl)
		self.msgs = self.extractor.read_msgs_from_file("./datasets/currentlyWorkingDataset/all_noTimestamps_sent_ENSEMBLE.csv")

		#0 format of self.msgs is a list of dicts, each dict equal to a row in the csv (i.e. a message) (i.e. a dict of the format {"uid": uid, "content": content, "sentiment": sentiment})

	def get_user_embedding(self, uid) -> torch.Tensor:
		"""
		This function retrieves or creates a user embedding tensor based on the user's messages.
		
		@param uid uid stands for "user ID", which is a unique identifier assigned to each user in a system
		or application. In this code, it is used to retrieve or create an embedding (a numerical
		representation) for a specific user based on their messages.
		
		@return a torch.Tensor object, which is the user embedding for the given user ID. If the user
		embedding has already been computed and stored in memory, it is returned directly. Otherwise, the
		function extracts features from the user's messages, creates a user embedding from these features,
		and stores the embedding in memory and on disk before returning it. If there are no messages for the
		given user ID
		"""
		path = os.path.join("./userEmbeddings", f"{uid}.pt")

		if os.path.exists(path):
			self.userEmbeddings[uid] = torch.load(path)
		else:
			if uid not in self.userEmbeddings:
				usrMsgs = [msg for msg in self.msgs if msg["uid"] == uid]

				with tqdm(total=len(usrMsgs), desc=f"Processing user {uid} embs") as pbar:
					usrMsgs = [msg for msg in usrMsgs if len(msg["content"]) >= 25]
					pbar.update()
					if not usrMsgs:
						print(f"no msgs for {uid}")
						self.userEmbeddings[uid] = torch.zeros(self.userEmbeddingSize) #1 NOT A SMART THING TO DO
					else:
						feats = self.extractor.extract_and_store_feats(usrMsgs, uid)
						pbar.update()
						if feats is not None:
							self.userEmbeddings[uid] = self.create_user_embedding(feats)
							torch.save(self.userEmbeddings[uid], path)
							pbar.update()
						else:
							print(f"None feats for {uid}")
							return torch.zeros(self.userEmbeddingSize) #1 NOT A SMART THING TO DO
		return self.userEmbeddings[uid]

	def update_user_embedding(self, uid) -> None:
		"""
		This function updates the user embedding for a given user ID by extracting features from messages
		and creating a user embedding.
		
		@param uid uid stands for "user ID". It is a unique identifier for a specific user in the system.
		The function `update_user_embedding` updates the embedding (a numerical representation) of the user
		with the given ID based on their message history.
		"""
		with tqdm(total=1, desc=f"Updating user {uid} emb") as pbar:
			feats = self.extractor.extract_and_store_feats(self.msgs, uid)
			pbar.update()

			self.userEmbeddings[uid] = self.create_user_embedding(feats)
			pbar.update()

	def create_user_embedding(self, feats) -> torch.Tensor:
		"""
		This function creates a user embedding tensor from a dictionary of features.
		
		@param feats The `feats` parameter is a dictionary containing features of a user, where the keys are
		feature names and the values are feature values. The function creates a user embedding by converting
		the feature values to a tensor of floats and returning it. If any of the feature values are `None`,
		the function
		
		@return a PyTorch tensor containing the numerical features passed as input to the function. If the
		input is empty or contains None values, the function will print an error message and return None.
		"""
		if not feats:
			print(f"not feats created!")
			return None

		for k, v in feats.items():
			if v is None:
				print(f"None value for {k}")
				raise ValueError
		feats = [v for v in feats.values() if isinstance(v, (int,float))]
		emb = torch.tensor(feats, dtype=torch.float)
		return emb