from typing import List, Dict
import torch


class ACARISCollator:
	def __call__(self, feats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		maxLen = max([len(feat["input_ids"]) for feat in feats])

		inputIDs = torch.zeros((len(feats), maxLen), dtype=torch.long)
		attentionMask = torch.zeros((len(feats), maxLen), dtype=torch.long)

		for idx, feat in enumerate(feats):
			inputIDs[idx, :len(feat["input_ids"])] = feat["input_ids"]
			attentionMask[idx, :len(feat["attention_mask"])] = feat["attention_mask"]

			if len(feat["input_ids"]) < maxLen:
				inputIDs[idx, len(feat["input_ids"]):] = torch.zeros(maxLen - len(feat["input_ids"]), dtype=torch.long)
				attentionMask[idx, len(feat["attention_mask"]):] = torch.zeros(maxLen - len(feat["attention_mask"]), dtype=torch.long)

		userEmbs = torch.stack([feat["userEmbedding"] for feat in feats])
		labels = torch.stack([feat["label"] for feat in feats])

		return {
			"input_ids": inputIDs,
			"attention_mask": attentionMask,
			"userEmbedding": userEmbs,
			"label": labels
		}