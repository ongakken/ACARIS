from torch.utils.data import Dataset
import torch
from preprocess import Preprocessor
from user_embedder import UserEmbedder


class ACARISDs(Dataset):
	def __init__(self, data, preprocessor, userEmbedder):
		self.data = data
		self.preprocessor = preprocessor
		self.userEmbedder = userEmbedder

		self.tokenized = [self.preprocessor.tokenize(text, maxLen=None, padding=False, truncation=False, returnTensors=None) for text in self.data["content"]]
		self.maxLen = 128

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

		tokens = self.preprocessor.tokenize(content, maxLen=self.maxLen)

		inputIDsPadded = torch.zeros(128, dtype=torch.long)
		attentionMaskPadded = torch.zeros(128, dtype=torch.long)

		inputIDsPadded[:len(tokens["input_ids"].squeeze())] = tokens["input_ids"].squeeze()
		attentionMaskPadded[:len(tokens["attention_mask"].squeeze())] = tokens["attention_mask"].squeeze()

		userEmbedding = self.userEmbedder.get_user_embedding(userID)

		print(f"input_ids shapes: {tokens['input_ids'].squeeze().shape}")
		print(f"attention_mask shapes: {tokens['attention_mask'].squeeze().shape}")
		print(f"userEmbedding shape: {userEmbedding.shape}")

		print(f"returning uid: {userID}")

		return {
			"input_ids": tokens["input_ids"].squeeze(),
			"attention_mask": tokens["attention_mask"].squeeze(),
			#"userEmbedding": userEmbedding,
			"uid": userID,
			"label": torch.tensor(label, dtype=torch.long)
		}
