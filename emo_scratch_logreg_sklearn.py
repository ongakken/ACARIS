import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import pickle

wandb.init(project="emo-scratch-RF-sklearn")

df = pd.read_csv("datasets/go_emotions_dataset.csv")
df.head()

emotion_counts = {}
for c in df.columns[3:-1]:
    emotion_counts[c]  = df[c].value_counts().to_dict()[1]
print("Emotion counts:", emotion_counts)
print("---------------------------------")
classes = list(df.columns[3:])
print("Emotions:", classes)
print("---------------------------------")

X_train, X_test, y_train, y_test = train_test_split(df["text"], df.iloc[:,3:], test_size=0.15, random_state=69)
print("Training set:", X_train.shape, y_train.shape)
print("y_train:\n", y_train)
print("Testing set:", X_test.shape, y_test.shape)

y_train = y_train.values.argmax(axis=1)
y_test = y_test.values.argmax(axis=1)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

## Set and log hyperparams ##
max_features = 99999
n_estimators = 150
max_depth = 1000
min_samples_split = 2
min_samples_leaf = 1
class_weight = "balanced"
algo = "RandomForestClassifier"
wandb.config.update({"x_train_len" : len(X_train), "x_test_len" : len(X_test), "y_train_len": len(y_train), "y_test_len": len(y_test), "max_features": max_features, "n_estimators": n_estimators, "max_depth": max_depth, "min_samples_split": min_samples_split, "min_samples_leaf": min_samples_leaf, "class_weight": class_weight, "algo": algo})


print("-------------- Extracting features --------------")
vect = TfidfVectorizer(max_features=max_features)
X_train_ft = vect.fit_transform(X_train)
X_test_ft = vect.transform(X_test)

print("-------------- Normalizing features --------------")
scaler = MaxAbsScaler()
X_train_ft = scaler.fit_transform(X_train_ft)
X_test_ft = scaler.transform(X_test_ft)

print("-------------- Training RF --------------")
RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, class_weight=class_weight, verbose=1, random_state=69)
RF.fit(X_train_ft, y_train)

print("-------------- Eval RF --------------")
y_pred = RF.predict(X_test_ft)

print("-------------- Calling W&B --------------")
wandb.log({"classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)})
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

print("-------------- Predicting with classifier --------------")
print(classes[RF.predict(vect.transform(["Oh my God, she's so cute!"]))[0]])

print("-------------- Saving model --------------")
pickle.dump(RF, open("models/RF.pkl", "wb"))
pickle.dump(vect, open("models/vect.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
print("-------------- Done --------------")

wandb.finish()