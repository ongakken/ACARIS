from transformers import DistilBertTokenizerFast
import torch
import sys
from trainDistilBERT import DistilBertForMulticlassSequenceClassification
import os
import numpy as np
import wandb

wandb.init(mode="disabled")
os.environ["WANDB_MODE"] = "disabled"

class InferACARISBERT:
	def __init__(self, mdlPath):
		self.mdlPath = mdlPath
		self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(self.mdlPath)

	def prepare_input(self, text):
		inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
		return inputs

	def predict(self, inputs):
		outputs = self.model(**inputs)
		probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
		return probs.detach().cpu().numpy()

	def infer(self):
		for text in sys.stdin:
			text = text.strip()
			inputs = self.prepare_input(text)
			pred = self.predict(inputs)
			print(f"Input: {text}\nPrediction: {pred}\n")
			maxProb = np.argmax(pred, axis=-1)
			label = self.model.config.id2label[maxProb[0]]
			confidence = pred[0][maxProb[0]]
			print(f"Label: {label}\nConfidence: {confidence}\n")

if __name__ == "__main__":
	acaris_bert_infer = InferACARISBERT("./output")
	acaris_bert_infer.infer()