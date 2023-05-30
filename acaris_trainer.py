from transformers import Trainer, TrainerCallback, EvalPrediction
import torch.nn as nn
import torch
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, classification_report
import wandb
from wandb import AlertLevel
from alert import send_alert


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



class ACARISCrossEntropyLoss(nn.Module):
	def __init__(self, nClasses):
		super(ACARISCrossEntropyLoss, self).__init__()
		self.nClasses = nClasses
		self.loss = nn.CrossEntropyLoss()

	def forward(self, logits, labels):
		mapped = (labels + 1) // 2
		return self.loss(logits, mapped)



class ACARISTrainer(Trainer):
	def __init__(self, model, trainLoader, evalLoader, progressCb=None, *args, **kwargs):
		super().__init__(model=model, *args, **kwargs)
		self.trainLoader = trainLoader
		self.evalLoader = evalLoader
		self.lossFct = ACARISCrossEntropyLoss(model.bert.config.num_labels)

	def get_train_dataloader(self):
		"""
		This function returns the training data loader if it exists, otherwise it calls the parent class's
		function to get the training data loader.
		
		@return If `self.trainLoader` is not `None`, then `self.trainLoader` is returned. Otherwise, the
		method `get_train_dataloader()` of the parent class is called and its output is returned.
		"""
		if self.trainLoader is not None:
			return self.trainLoader
		else:
			return super().get_train_dataloader()

	def get_eval_dataloader(self, *args, **kwargs):
		"""
		This function returns the evaluation dataloader if it exists, otherwise it calls the parent class's
		get_eval_dataloader method.
		
		@return If `self.evalLoader` is not `None`, then it returns `self.evalLoader`. Otherwise, it calls
		the `get_eval_dataloader` method of the parent class (using `super()`) and returns its output.
		"""
		if self.evalLoader is not None:
			return self.evalLoader
		else:
			return super().get_eval_dataloader(*args, **kwargs)

	def train(self):
		"""
		The `train` function is called on the parent class using `super()` in a subclass.
		"""
		super().train()


	def compute_loss(self, model, inputs, return_outputs=False):
		"""
		This function computes the loss for a given model and inputs, and returns the loss value and outputs
		if specified.
		
		@param model The neural network model being used for training or inference.
		@param inputs a dictionary containing the input data for the model, including the input ids,
		attention masks, and token type ids.
		@param return_outputs A boolean parameter that determines whether to return only the loss value or
		both the loss value and the model outputs.
		
		@return The function `compute_loss` returns either the loss value or a tuple containing the loss
		value and the outputs, depending on the value of the `return_outputs` parameter.
		"""
		uids = inputs.get("uids")
		labels = inputs.get("labels")
		if uids is not None:
			pass
		else:
			raise ValueError("uids is None in compute_loss")
		
		inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "uids"}

		outputs = model(**inputs, uids=uids)
		logits = outputs["logits"]

		if labels is not None and uids is not None:
			labels = torch.clamp(labels, min=0, max=model.bert.config.num_labels - 1)
			loss = self.lossFct(logits, labels)
		else:
			raise ValueError("labels or uids is None")

		return (loss, outputs) if return_outputs else loss

	# def compute_metrics(self, evalPred: EvalPrediction):
	# 	logits, labels = evalPred.predictions, evalPred.label_ids
	# 	predictions = torch.argmax(logits, dim=-1)

	# 	if len(labels) > 0:
	# 		acc = (predictions == labels).sum().item() / len(labels)
	# 	else:
	# 		acc = 0
			
	# 	wandb.log({"eval_accuracy": acc})
	# 	metrics = {"eval_accuracy": acc}
	# 	print(f"metrics: {metrics}")
	# 	return metrics

	def prediction_step(self, mdl, inputs: dict[str, torch.Tensor], prediction_loss_only, ignore_keys=None):
		"""
		This function performs a prediction step using a given model and inputs, and returns the loss,
		logits, and labels (if available).
		
		@param mdl The model being used for prediction.
		@param inputs a dictionary containing the inputs to the model, including the input ids, attention
		masks, and token type ids
		@param prediction_loss_only A boolean indicating whether to only return the prediction loss or the
		full output. If True, only the loss will be returned. If False, the loss, logits, and labels (if
		available) will be returned.
		@param ignore_keys `ignore_keys` is a tuple of keys that should be ignored when extracting the
		logits from the `outputs` dictionary. These keys are typically used for auxiliary outputs that are
		not used for computing the final loss.
		
		@return a tuple containing different values depending on whether labels are provided or not. If
		labels are provided, the tuple contains the loss, logits, and labels. If labels are not provided,
		the tuple contains only the logits. If the argument `prediction_loss_only` is True, the tuple
		contains only the loss and no logits or labels.
		"""
		labels = inputs.get("labels")
		labels = torch.tensor(labels)
		labels = labels.to(mdl.device)
		inputs = self._prepare_inputs(inputs)

		inputsCopy = inputs.copy()
		input_ids = inputsCopy.pop("input_ids")
		attention_mask = inputsCopy.pop("attention_mask")
		uids = inputsCopy.pop("uids")
		labels = inputsCopy.pop("labels") if "labels" in inputsCopy else None

		with torch.no_grad():
			if uids is not None:
				pass
			else:
				wandb.alert(title="uids is None", text="uids is None", level=AlertLevel.ERROR)
				send_alert("TRAINING WARNING", "uids is None", "normal", 60000)
			outputs = mdl(input_ids=input_ids, attention_mask=attention_mask, labels=labels, uids=uids) # the two asterisks (**) unpack the dictionary
			if outputs is None:
				wandb.alert(title="outputs is None", text="outputs is None", level=AlertLevel.ERROR)
				send_alert("TRAINING ERROR", "outputs is None", "critical", 60000)
				raise ValueError("outputs is None")
			if labels is not None:
				loss, _ = self.compute_loss(mdl, inputs, return_outputs=True)
				print(f"loss: {loss}")
				if loss is None:
					wandb.alert(title="loss is None", text="loss is None", level=AlertLevel.ERROR)
					raise ValueError("loss is None")
				loss = loss.mean().detach()
				if self.args.prediction_loss_only:
					return (loss, None, None)

		if outputs is None:
			return (None, None, None)
		if ignore_keys is not None:
			logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
		else:
			logits = tuple(v for k, v in outputs.items())
		if len(logits) == 1:
			logits = logits[0]

		if labels is not None:
			#labels = tuple(inputs.get(name).detach() for name in labels)
			#labels = (inputs.get(name).detach() if inputs.get(name) is not None else None for name in labels)
			if labels is None:
				wandb.alert(title="labels is None", text="labels is None", level=AlertLevel.ERROR)
				raise ValueError("labels is None")
			return (loss, logits, labels)
		else:
			return (logits,)