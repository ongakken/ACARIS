from transformers import Trainer, TrainerCallback, EvalPrediction
import torch.nn as nn
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score



class ProgressCb(TrainerCallback):
	def __init__(self):
		super().__init__()

	def on_epoch_begin(self, args, state, control, **kwargs):
		print(f"Epoch {state.epoch +1}")
		self.progressBar = tqdm(total=state.max_steps, desc="Training", leave=True)

	def on_step_end(self, args, state, control, **kwargs):
		if state.log_history:
			lastLog = state.log_history[-1]
			if "loss" in lastLog:
				self.progressBar.update(1)
				print(f"Loss on step end: {lastLog['loss']}")

	def on_epoch_end(self, args, state, control, **kwargs):
		self.progressBar.close()



class ACARISTrainer(Trainer):
	def __init__(self, trainLoader, evalLoader, progressCb=None, *args, **kwargs):
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

	def train(self):
		super().train()


	def compute_loss(self, model, inputs, labels=None, return_outputs=False):
		labels = inputs.pop("labels", None)
		userEmbedding = inputs.pop("userEmbedding", None)
		if userEmbedding is None or labels is None:
			return (None, None) if return_outputs else None
		
		inputs = {k: v.to(model.device) for k, v in inputs.items()}
		userEmbedding = userEmbedding.to(model.device)

		outputs = model(**inputs, userEmbedding=userEmbedding)
		logits = outputs["logits"]

		print(f"inputs: {inputs}")
		print(f"logits: {logits}")

		if labels is not None:
			# print(f"labels.shape: {labels.shape}")
			# print(f"logits.shape: {logits.shape}")

			lossFct = nn.CrossEntropyLoss()
			labels = torch.clamp(labels, min=0, max=model.bert.config.num_labels - 1)
			loss = lossFct(logits, labels)
		else:
			loss = None

		return (loss, outputs) if return_outputs else loss

	def compute_metrics(self, evalPred: EvalPrediction):
		logits, labels = evalPred.predictions, evalPred.label_ids
		predictions = torch.argmax(logits, dim=-1)

		if len(labels) > 0:
			acc = (predictions == labels).sum().item() / len(labels)
		else:
			acc = 0
			
		wandb.log({"eval_accuracy": acc})
		metrics = {"eval_accuracy": acc}
		print(f"metrics: {metrics}")
		return metrics

	def prediction_step(self, mdl, inputs, prediction_loss_only, ignore_keys=()):
		labels = inputs.pop("labels", None)
		inputs = self._prepare_inputs(inputs)

		with torch.no_grad():
			userEmbs = inputs.pop("userEmbedding", None)
			if userEmbs is not None:
				userEmbs = userEmbs.to(inputs["input_ids"].device)
			outputs = mdl(**inputs, userEmbedding=userEmbs)
			if outputs is None:
				return (None, None, None)
			print(f"mdl output: {outputs}")
			if labels is not None:
				print(f"labels before loss: {labels}")
				loss, _ = self.compute_loss(mdl, inputs, labels=labels, return_outputs=True)
				print(f"loss: {loss}")
				if loss is None:
					#return (None, None, None)
					loss = torch.tensor(1e6).to(mdl.device)
				loss = loss.mean().detach()
				if self.args.prediction_loss_only:
					return (loss, None, None)

		logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
		if len(logits) == 1:
			logits = logits[0]

		if labels is not None:
			labels = tuple(inputs.get(name).detach() for name in labels)
			if len(labels) == 1:
				labels = labels[0]
			return (loss, logits, labels)
		else:
			return (logits,)