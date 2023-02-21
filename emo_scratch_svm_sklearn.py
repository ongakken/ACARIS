import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import winsound
import time

def ckpt(freq=800, duration=0.33, sleep=0, reps=1):
    duration *= 1000
    for i in range(reps):
        winsound.Beep(freq, int(duration))
        time.sleep(sleep)

wandb.init(project="emo-scratch-svm-sklearn")

df = pd.read_csv("dataset/go_emotions_dataset.csv")
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

### Set and log hyperparams ###
kernel = "linear"
C = 0.9
tol = 0.0001
gamma = "scale"
max_iter = 1000
max_features = 300000
b_linearSVC = True
multi_class = "ovr"
wandb.config.update({"x_train_len" : len(X_train), "x_test_len" : len(X_test), "y_train_len": len(y_train), "y_test_len": len(y_test), "kernel" : kernel, "C" : C, "tol" : tol, "max_iter" : max_iter, "max_features" : max_features, "b_linearSVC" : b_linearSVC, "multi_class" : multi_class})

print("-------------- Extracting features --------------")
#vect = TfidfVectorizer(stop_words="english", max_features=max_features)
vect = TfidfVectorizer(max_features=max_features)
X_train_ft = vect.fit_transform(X_train)
X_test_ft = vect.transform(X_test)

# print("-------------- Scaling features --------------")
# scaler = MaxAbsScaler()
# X_train_ft = scaler.fit_transform(X_train_ft)
# X_test_ft = scaler.transform(X_test_ft)

print("-------------- Training SVM --------------")
ckpt(reps=1, sleep=0.5)
if not b_linearSVC:
    svm = SVC(verbose=True, max_iter=max_iter, kernel=kernel, tol=tol, C=C, gamma=gamma)
else:
    svm = LinearSVC(verbose=True, max_iter=max_iter, tol=tol, C=C, multi_class=multi_class)
svm.fit(X_train_ft, y_train)
ckpt(reps=2, sleep=0.5)

print("-------------- Eval SVM --------------")
y_pred = svm.predict(X_test_ft)
print(classification_report(y_test, y_pred, target_names=classes))
ckpt(reps=3, sleep=0.5)

print("-------------- Calling W&B --------------")
wandb.log({"classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)})

print("-------------- Predicting with classifier --------------")
print(classes[svm.predict(vect.transform(["Oh my God, she's so cute!"]))[0]])

#print("-------------- Saving model --------------")

wandb.finish()