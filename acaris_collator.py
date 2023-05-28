from typing import List, Dict
import torch


class ACARISCollator:
	def __init__(self, device, userEmbedder):
		self.device = device
		self.userEmbedder = userEmbedder

	def __call__(self, feats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		"""
		This function takes in a list of feature dictionaries, pads the input_ids and attention_mask tensors
		to the maximum length, stacks the labels tensor, and returns a dictionary containing the padded
		input_ids and attention_mask tensors, uids, and stacked labels tensor.
		
		@param feats A list of dictionaries, where each dictionary contains the input_ids, attention_mask,
		uids, and labels for a single example in a dataset.
		
		@return a dictionary containing the following keys and their corresponding values:
		- "input_ids": a tensor of shape (batch_size, maxLen) containing the input IDs of the input
		sequences
		- "attention_mask": a tensor of shape (batch_size, maxLen) containing the attention masks of the
		input sequences
		- "uids": a list of unique user identifiers for each input sequence
		"""
		maxLen = max([len(feat["input_ids"]) for feat in feats])

		inputIDs = torch.zeros((len(feats), maxLen), dtype=torch.long)
		attentionMask = torch.zeros((len(feats), maxLen), dtype=torch.long)

		for idx, feat in enumerate(feats):
			inputIDs[idx, :len(feat["input_ids"])] = feat["input_ids"]
			attentionMask[idx, :len(feat["attention_mask"])] = feat["attention_mask"]

			if len(feat["input_ids"]) < maxLen:
				inputIDs[idx, len(feat["input_ids"]):] = torch.zeros(maxLen - len(feat["input_ids"]), dtype=torch.long)
				attentionMask[idx, len(feat["attention_mask"]):] = torch.zeros(maxLen - len(feat["attention_mask"]), dtype=torch.long)

		uids = [feat["uids"] for feat in feats]
		labels = torch.stack([feat["labels"] for feat in feats])

		return {
			"input_ids": inputIDs,
			"attention_mask": attentionMask,
			"uids": uids,
			"labels": labels
		}