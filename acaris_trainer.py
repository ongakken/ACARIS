from transformers import Trainer
import torch.nn as nn


class ACARISTrainer(Trainer):
	def __init__(self, trainLoader, evalLoader, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.trainLoader = trainLoader
		self.evalLoader = evalLoader

	def get_train_dataloader(self):
		if self.trainLoader is not None:
			return self.trainLoader
		else:
			return super().get_train_dataloader()

	def get_eval_dataloader(self, *args, **kwargs):
		if self.evalLoader is not None:
			return self.evalLoader
		else:
			return super().get_eval_dataloader(*args, **kwargs)


	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.pop("label").to(inputs["input_ids"].device)
		userIDs = inputs.pop("uid")
		print(f"recv userIDs: {userIDs}")
		outputs = model(**inputs, userIDs=userIDs)
		logits = outputs["logits"]

		lossFct = nn.CrossEntropyLoss()
		loss = lossFct(logits.view(-1, model.bert.config.num_labels), labels.view(-1))

		return (loss, outputs) if return_outputs else loss