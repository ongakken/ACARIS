from torch.utils.data import Dataset
import torch
from preprocess import Preprocessor
from user_embedder import UserEmbedder


class ACARISDs(Dataset):
	def __init__(self, data, preprocessor, userEmbedder):
		self.data = data
		self.preprocessor = preprocessor
		self.userEmbedder = userEmbedder

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = self.data[idx]
		text = sample["text"]
		label = sample["label"]
		userID = sample["userID"]

		tokens = self.preprocessor.tokenize(text, maxLen=64, padding=True, truncation=True)
		userEmbedding = self.userEmbedder.get_user_embedding(userID)

		return {
			"input_ids": tokens["input_ids"].squeeze(),
			"attention_mask": tokens["attention_mask"].squeeze(),
			"userEmbedding": userEmbedding,
			"label": torch.tensor(label, dtype=torch.long)
		}
