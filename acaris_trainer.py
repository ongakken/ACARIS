from transformers import Trainer, TrainerCallback
import torch.nn as nn
import torch
from tqdm.auto import tqdm



class ProgressCb(TrainerCallback):
	def __init__(self):
		super().__init__()

	def on_epoch_begin(self, args, state, control, **kwargs):
		print(f"Epoch {state.epoch +1}")
		self.progressBar = tqdm(total=state.max_steps, desc="Training", leave=True)

	def on_step_end(self, args, state, control, **kwargs):
		self.progressBar.update(1)
		print(f"Loss: {state.loss}")

	def on_epoch_end(self, args, state, control, **kwargs):
		self.progressBar.close()



class ACARISTrainer(Trainer):
	def __init__(self, trainLoader, evalLoader, progressCb=None, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.trainLoader = trainLoader
		self.evalLoader = evalLoader
		self.progressCb = ProgressCb

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

	def train(self):
		if self.progressCb is not None:
			self.progressCb = self.progressCb()
		super().train()


	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.pop("labels", None)
		userEmbedding = inputs.pop("userEmbedding", None)
		if userEmbedding is None or labels is None:
			return (None, None) if return_outputs else None
		userEmbedding = userEmbedding.to(inputs["input_ids"].device)
		outputs = model(**inputs, userEmbedding=userEmbedding)
		logits = outputs["logits"]

		if labels is not None:
			print(f"labels.shape: {labels.shape}")
			print(f"logits.shape: {logits.shape}")

			lossFct = nn.CrossEntropyLoss()
			labels = torch.clamp(labels, min=0, max=model.bert.config.num_labels - 1)
			loss = lossFct(logits, labels)
		else:
			loss = None

		return (loss, outputs) if return_outputs else loss

	def prediction_step(self, mdl, inputs, prediction_loss_only, ignore_keys=()):
		hasLabels = "labels" in inputs
		inputs = self._prepare_inputs(inputs)

		if hasLabels:
			labels = inputs.pop("labels", None)
			if labels is not None:
				labels = labels.to(inputs["input_ids"].device)

		with torch.no_grad():
			userEmbs = inputs.pop("userEmbedding", None)
			if userEmbs is not None:
				userEmbs = userEmbs.to(inputs["input_ids"].device)
			outputs = mdl(**inputs, userEmbedding=userEmbs)
			if hasLabels:
				loss, _ = self.compute_loss(mdl, inputs, return_outputs=True)
				if loss is None:
					return (None, None, None)
				loss = loss.mean().detach()
				if self.args.prediction_loss_only:
					return (loss, None, None)

		logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
		if len(logits) == 1:
			logits = logits[0]

		if hasLabels:
			labels = tuple(inputs.get(name).detach() for name in labels)
			if len(labels) == 1:
				labels = labels[0]
			return (loss, logits, labels)
		else:
			return (logits,)