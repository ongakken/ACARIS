from transformers import DistilBertTokenizerFast
import sys
from trainACARIS import DistilBertForMulticlassSequenceClassification
import os

class InferACARISBERT:
	def __init__(self, mdlPath):
		self.mdlPath = mdlPath
		self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
		self.model = DistilBertForMulticlassSequenceClassification.from_pretrained(self.mdlPath)

	def push(self, modelName, org="ongknsro"):
		self.model.push_to_hub(repo_id=f"{org}/{modelName}", private=True)
		self.tokenizer.push_to_hub(repo_id=f"{org}/{modelName}")

if __name__ == "__main__":
	acaris_bert_infer = InferACARISBERT("./output/checkpoint-3540")
	acaris_bert_infer.push("ACARIS-DistilBERT-iter1-fp16-batchSize14")