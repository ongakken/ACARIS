from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import label_binarize
from inferDistilBERT import InferACARISBERT
from inferACARIS import InferACARIS
import pandas as pd
import numpy as np
from tqdm import tqdm



class ACARISComparer:
	def __init__(self, baseline, acarisModel):
		self.baseline = baseline
		self.acarisModel = acarisModel

	def compare(self, data):
		uids = list(data["uid"])
		contents = list(data["content"])
		classLabels = ["neg", "neu", "pos"]
		y_true = [classLabels.index(sentiment) for sentiment in data["sentiment"]]

		baselineIN = [self.baseline.prepare_input(content) for content in contents]

		baselinePreds = [self.baseline.predict(inputs=input_) for input_ in tqdm(baselineIN, desc="Predicting using baseline", ncols=100)]
		acarisPreds = [self.acarisModel.predict(uids=[uid], contents=[content]) for uid, content in tqdm(zip(uids, contents), total=len(uids), desc="Predicting using ACARIS", ncols=100)]

		probs1 = baselinePreds
		probs2 = np.vstack([pred[1] for pred in acarisPreds])
		y_pred1 = [np.argmax(prob) for prob in probs1]
		y_pred2 = [np.argmax(prob) for prob in probs2]

		print("Baseline metrics:")
		y_true = np.array(y_true)
		y_pred1 = np.array(y_pred1)
		probs1 = np.concatenate(probs1)
		self.compute_metrics(y_true, y_pred1, probs1, classLabels)

		print("ACARIS metrics:")
		self.compute_metrics(y_true, y_pred2, probs2, classLabels)

	def compute_metrics(self, y_true, y_pred, y_probs, classLabels):
		print(classification_report(y_true, y_pred, target_names=classLabels))

		confMatrix = confusion_matrix(y_true, y_pred)
		print(confMatrix)

		y_true_bin = label_binarize(y_true, classes=range(len(classLabels)))
		roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr")
		print(f"ROC AUC: {roc_auc}")

		logloss = log_loss(y_true, y_probs)
		print(f"Log loss: {logloss}")


if __name__ == "__main__":
	baseline = InferACARISBERT("ongknsro/ACARIS_BASELINE-DistilBERT-evalSpecimen1-batchSize32")
	ACARISModel = InferACARIS("ongknsro/ACARIS-DistilBERT_MLPUserEmbs-iter1-batchSize32")

	comparer = ACARISComparer(baseline, ACARISModel)

	data = pd.read_csv("./datasets/test.csv", sep="|", encoding="utf-8")
	comparer.compare(data)