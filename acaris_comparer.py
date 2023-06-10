from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, log_loss, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize
from inferDistilBERT import InferACARISBERT
from inferACARIS import InferACARIS
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import probplot
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from termcolor import colored
import os

classLabels = ["neg", "neu", "pos"]



class ACARISComparer:
	def __init__(self, baseline, acarisModel, baselineName, acarisName):
		self.baseline = baseline
		self.acarisModel = acarisModel
		self.baselineName = baselineName
		self.acarisName = acarisName
		self.resDir = f"conclusions/{baselineName}_vs_{acarisName}"
		os.makedirs(self.resDir, exist_ok=True)

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
		baselineStat = self.compute_metrics(y_true, y_pred1, probs1, classLabels)

		print("ACARIS metrics:")
		acarisStat = self.compute_metrics(y_true, y_pred2, probs2, classLabels)

		print("Statistical tests:")
		statRes = self.stat_test(baselineStat, acarisStat)

		with open(f"{self.resDir}/conclusions.txt", "w") as f:
			print(colored("-------------------------\n\n\n\nCONCLUSION\n\n\n\n-------------------------", "yellow"))
			significant = [metric for metric, p, in statRes if p < 0.05]
			improved = [metric for metric in significant if acarisStat[metric] > baselineStat[metric] * 1.05]

			f.write("Conclusion:\n\n")
			if significant:
				print(colored(f"The ACARIS model exhibits statistically significant improvement over the baseline model on the following metrics: {', '.join(significant)}\n", "green"))
				f.write(f"The ACARIS model exhibits statistically significant improvement over the baseline model on the following metrics: {', '.join(significant)}\n")
			else:
				print(colored("The ACARIS model does NOT exhibit statistically significant improvement over the baseline model on any metric.", "red"))
				f.write("The ACARIS model does not exhibit statistically significant improvement over the baseline model on any metric.\n")

			if improved:
				print(colored(f"The ACARIS model exhibits practically significant improvement over the baseline model on the following metrics: {', '.join(improved)}", "green"))
				f.write(f"The ACARIS model exhibits practically significant improvement over the baseline model on the following metrics: {', '.join(improved)}\n")
			else:
				print(colored("The ACARIS model does NOT exhibit practically significant improvement over the baseline model on any metric.", "red"))
				f.write("The ACARIS model does NOT exhibit practically significant improvement over the baseline model on any metric.\n")

	def compute_metrics(self, y_true, y_pred, y_probs, classLabels):
		print(classification_report(y_true, y_pred, target_names=classLabels))

		confMatrix = confusion_matrix(y_true, y_pred)
		print(confMatrix)

		acc = accuracy_score(y_true, y_pred)

		y_true_bin = label_binarize(y_true, classes=range(len(classLabels)))
		roc_auc = roc_auc_score(y_true_bin, y_probs, multi_class="ovr")
		print(f"ROC AUC: {roc_auc}")

		logloss = log_loss(y_true, y_probs)
		print(f"Log loss: {logloss}")

		precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
		
		metrics = {
			"precision": precision,
			"recall": recall,
			"f1": f1,
			"roc_auc": roc_auc,
			"logloss": logloss,
			"confusion_matrix": confMatrix,
			"accuracy": acc
		}

		return metrics
		

	def stat_test(self, metrics1, metrics2):
		res = []
		for metric, values1 in metrics1.items():
			values2 = metrics2[metric]

			if isinstance(values1, np.ndarray):
				if metric == "confusion_matrix":
					plt.figure(figsize=(10, 4))
					plt.subplot(1, 2, 1)
					sns.heatmap(values1, annot=True, cmap="Blues", fmt="d", xticklabels=classLabels, yticklabels=classLabels)
					plt.title("Baseline")
					plt.subplot(1, 2, 2)
					sns.heatmap(values2, annot=True, cmap="Reds", fmt="d", xticklabels=classLabels, yticklabels=classLabels)
					plt.title("ACARIS")
					plt.tight_layout()
					plt.savefig(f"{self.resDir}/{metric}.png")
					plt.show()
				else:
					print(f"Visual check on distro of {metric}")
					plt.figure(figsize=(10, 5))
					# hist for normality
					plt.subplot(1, 2, 1)
					sns.histplot(values1, color="red", label="Baseline", kde=True)
					sns.histplot(values2, color="blue", label="ACARIS", kde=True)
					plt.legend()
					plt.title(f"Histogram for {metric}")
					# qqplot for normality
					plt.subplot(2, 2, 2)
					probplot(values1, plot=plt)
					plt.title(f"QQ plot (baseline) for {metric}")
					plt.subplot(2, 2, 4)
					probplot(values2, plot=plt)
					plt.title(f"QQ plot (ACARIS) for {metric}")
					plt.tight_layout()
					plt.savefig(f"{self.resDir}/{metric}.png")
					plt.show()
					shapiro1 = stats.shapiro(values1)[1]
					shapiro2 = stats.shapiro(values2)[1]
					standardErr1 = np.std(values1) / np.sqrt(len(values1))
					standardErr2 = np.std(values2) / np.sqrt(len(values2))
					meanDiff = np.mean(values1) - np.mean(values2)
					print(f"STDs: {np.std(values1)}, {np.std(values2)}")
					if np.abs(meanDiff) > 2 * (standardErr1 + standardErr2):
						print(colored(f"Mean difference is {meanDiff}, which is > 2x the standard error of the difference ({standardErr1 + standardErr2}).", "green"))
					else:
						print(colored(text=f"Mean difference is {meanDiff}, which is < 2x the standard error of the difference ({standardErr1 + standardErr2}).", color="red"))
					if shapiro1 > 0.05 and shapiro2 > 0.05:
						t, p = stats.ttest_rel(values1, values2)
						print(f"Paired t-test on {metric}: t={t}, p={p}")
						test = "Paired t-test"
					else:
						if min(shapiro1, shapiro2) > 0.01:
							print(colored(f"Shapiro-Willk p-value is <0.01; 0.05>. Might not be reliably normally distributed.", "red"))
						w, p = stats.wilcoxon(values1, values2)
						test = "Wilcoxon signed-rank test"
					print(f"{test} on {metric}: stat={t if test=='Paired t-test' else w}, p={p}")
					if np.any(p < 0.05):
						print(colored(f"Significant diff on {metric}", "green"))
					else:
						print(colored(f"No significant diff on {metric}", "red"))
					res.append((metric, p))
			else:
				print(f"Metric {metric} diff: {values1 - values2}")
		return res

if __name__ == "__main__":
	baselineName = "ACARIS_BASELINE-DistilBERT-evalSpecimen1-batchSize32"
	acarisName = "ACARIS-DistilBERT_MLPUserEmbs-iter1-batchSize32"
	baseline = InferACARISBERT(f"ongknsro/{baselineName}")
	ACARISModel = InferACARIS(f"ongknsro/{acarisName}")

	comparer = ACARISComparer(baseline, ACARISModel, baselineName, acarisName)

	data = pd.read_csv("./datasets/test.csv", sep="|", encoding="utf-8")
	comparer.compare(data)