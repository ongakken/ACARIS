import torch
import torch.nn as nn
from transformers import AutoModel
from user_embedder import UserEmbedder
from acaris_trainer import ACARISCrossEntropyLoss


class ACARISMdl(nn.Module):
	def __init__(self, mdl, userEmbedder, numLabels = 3):
		super().__init__()
		self.bert = AutoModel.from_pretrained(mdl)
		self.classifier = nn.Linear(self.bert.config.hidden_size + 11, numLabels)
		self.userEmbedder = userEmbedder
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.lossFct = ACARISCrossEntropyLoss(self.bert.config.num_labels)
		self.to(self.device)

	def forward(self, input_ids, attention_mask, uids, labels=None):
		"""
		This function takes input_ids, attention_mask, and uids as inputs, and returns logits after
		concatenating token and user embeddings.
		
		@param input_ids The input sequence of token ids for the BERT model.
		@param attention_mask The attention mask is a binary tensor indicating which tokens in the input
		sequence are padding tokens and which are not. Padding tokens are usually added to ensure that all
		sequences in a batch have the same length. The attention mask is used by the transformer model to
		ignore the padding tokens during computation.
		@param uids The `uids` parameter is a list of user ids corresponding to the input sequence. It is
		used to retrieve user embeddings from a user embedding matrix using a user embedding module
		(`self.userEmbedder`). These user embeddings are then concatenated with the token embeddings
		obtained from the BERT model to form the final input
		@param labels The ground truth labels for the input data. If provided, the function will calculate
		and return the loss along with the logits. If not provided, only the logits will be returned.
		
		@return If `labels` is not None, a dictionary with keys "loss" and "logits" is returned, where
		"loss" is the calculated loss and "logits" are the predicted logits. If `labels` is None, a
		dictionary with only the "logits" key is returned, containing only the predicted logits.
		"""
		bertOut = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		tokenEmbs = bertOut.last_hidden_state[:, 0, :]
		userEmbs = [self.userEmbedder.get_user_embedding(uid) for uid in uids]
		userEmbs = torch.stack(userEmbs)
		userEmbs = userEmbs.to(tokenEmbs.device)
		combinedEmbs = torch.cat((tokenEmbs, userEmbs), dim=-1)
		logits = self.classifier(combinedEmbs) # logits are the raw, unnormalized predictions that a classification model generates. The softmax function is often used to convert logits to probabilities. The probabilities describe the likelihood of the input example belonging to each class.

		if labels is not None:
			loss = self.lossFct(logits, labels)
			return {"loss": loss, "logits": logits}
		else:
			return {"logits": logits}
