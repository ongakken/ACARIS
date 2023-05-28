from torch.utils.data import Dataset
import torch
from preprocess import Preprocessor
from user_embedder import UserEmbedder
from tqdm import tqdm


class ACARISDs(Dataset):
	def __init__(self, data, preprocessor, userEmbedder):
		self.data = data
		self.preprocessor = preprocessor
		self.userEmbedder = userEmbedder
		self.data = self.data.iloc[1:] # remove zeroth row (header)
		self.data = self.data[self.data["uid"].apply(lambda uid: isinstance(self.userEmbedder.get_user_embedding(uid), torch.Tensor))] # filter out users with no userEmbedding

		self.maxLen = 512

	def __len__(self) -> int:
		return len(self.data)

	def __getitem__(self, idx: int) -> dict:
		"""
		This function returns a dictionary containing tokenized input, attention mask, unique IDs, and
		labels for a given index in a dataset.
		
		@param idx idx is an integer representing the index of the sample to retrieve from the dataset.
		
		@return A dictionary containing the input IDs, attention mask, unique IDs, and labels of a sample
		from the dataset.
		"""
		sentMapping = {"pos": 2, "neg": 0, "neu": 1} # adjusted due to CrossEntropyLoss
		try:
			if idx < 0 or idx >= len(self.data):
				raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}")
		except TypeError:
			raise TypeError(f"Index {idx} is not an integer")
		sample = self.data.iloc[idx]
		content = sample["content"]
		label = sentMapping[sample["sentiment"]]
		try:
			uids = sample["uid"]
		except Exception as e:
			print(sample)
			raise

		tokens = self.preprocessor.tokenize(content, padding=True, truncation=True, maxLen=self.maxLen, returnTensors="pt")

		inputIDsPadded = torch.zeros(self.maxLen, dtype=torch.long)
		attentionMaskPadded = torch.zeros(self.maxLen, dtype=torch.long)

		inputIDsPadded[:len(tokens["input_ids"].squeeze())] = tokens["input_ids"].squeeze()
		attentionMaskPadded[:len(tokens["attention_mask"].squeeze())] = tokens["attention_mask"].squeeze()

		if uids is None:
			raise ValueError(f"UIDs for sample {idx} are None")

		return {
			"input_ids": inputIDsPadded,
			"attention_mask": attentionMaskPadded,
			"uids": uids,
			"labels": torch.tensor(label, dtype=torch.long)
		}
