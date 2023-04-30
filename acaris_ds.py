from torch.utils.data import Dataset
import torch
from preprocess import Preprocessor
from user_embedder import UserEmbedder


class ACARISDs(Dataset):
	def __init__(self, data, preprocessor, userEmbedder):
		self.data = data
		self.preprocessor = preprocessor
		self.userEmbedder = userEmbedder

		self.data = self.data[self.data["uid"].apply(lambda uid: self.userEmbedder.get_user_embedding(uid) is not None)]
		print(f"Number of samples after filtering: {len(self.data)}")

		self.tokenized = [self.preprocessor.tokenize(text, padding=False, truncation=False, returnTensors=None) for text in self.data["content"]]
		self.maxLen = 512

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sentMapping = {"pos": 1, "neg": -1, "neu": 0}
		if idx < 0 or idx >= len(self.data):
			raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}")
		sample = self.data.iloc[idx]
		content = sample["content"]
		label = sentMapping[sample["sentiment"]]
		try:
			userID = sample["uid"]
		except KeyError:
			print(f"KeyError at index {idx} with text {content} and label {label}")
			raise

		tokens = self.preprocessor.tokenize(content)

		inputIDsPadded = torch.zeros(self.maxLen, dtype=torch.long)
		attentionMaskPadded = torch.zeros(self.maxLen, dtype=torch.long)

		inputIDsPadded[:len(tokens["input_ids"].squeeze())] = tokens["input_ids"].squeeze()
		attentionMaskPadded[:len(tokens["attention_mask"].squeeze())] = tokens["attention_mask"].squeeze()

		userEmbedding = self.userEmbedder.get_user_embedding(userID)

		print(f"input_ids shapes: {tokens['input_ids'].squeeze().shape}")
		print(f"attention_mask shapes: {tokens['attention_mask'].squeeze().shape}")
		print(f"userEmbedding shape: {userEmbedding.shape}")

		return {
			"input_ids": inputIDsPadded,
			"attention_mask": attentionMaskPadded,
			"userEmbedding": userEmbedding,
			"label": torch.tensor(label, dtype=torch.long)
		}
