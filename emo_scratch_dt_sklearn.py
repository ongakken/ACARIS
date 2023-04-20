import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
import pandas as pd
import time

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

criterion = "gini"
max_depth = 500
max_leaf_nodes = 100
min_samples_split = 3
min_samples_leaf = 1
splitter = "random"
max_features = 300000
class_weight = "balanced"
algo = "DecisionTreeClassifier"

max_depth_range = range(100, 1000, 200)
max_leaf_nodes_range = range(10, 130, 30)
min_samples_split_range = range(2, 12, 3)
min_samples_leaf_range = range(1, 11, 1)
max_features_range = range(10000, 310000, 60000)

for i in max_depth_range:
    for j in max_leaf_nodes_range:
        for k in min_samples_split_range:
            for l in min_samples_leaf_range:
                for n in max_features_range:

                    wandb.init(project="emo-scratch-dt-sklearn")
                    
                    ### Set and log hyperparams ###
                    wandb.config.update({"x_train_len" : len(X_train), "x_test_len" : len(X_test), "y_train_len": len(y_train), "y_test_len": len(y_test), "max_features" : max_features, "criterion" : criterion, "max_depth" : max_depth, "max_leaf_nodes" : max_leaf_nodes, "min_samples_split" : min_samples_split, "min_samples_leaf" : min_samples_leaf, "splitter" : splitter, "class_weight" : class_weight, "algo" : algo})

                    print("-------------- Extracting features --------------")
                    vect = TfidfVectorizer(max_features=max_features)
                    X_train_ft = vect.fit_transform(X_train)
                    X_test_ft = vect.transform(X_test)

                    # print("-------------- Scaling features --------------")
                    # scaler = MaxAbsScaler()
                    # X_train_ft = scaler.fit_transform(X_train_ft)
                    # X_test_ft = scaler.transform(X_test_ft)

                    print("-------------- Training DT --------------")
                    dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, splitter=splitter, class_weight=class_weight, random_state=69)
                    dt.fit(X_train_ft, y_train)

                    print("-------------- Eval DT --------------")
                    y_pred = dt.predict(X_test_ft)

                    print("-------------- Calling W&B --------------")
                    wandb.log({"classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)})

                    print("-------------- Predicting with classifier --------------")
                    print(classes[dt.predict(vect.transform(["Oh my God, she's so cute!"]))[0]])

                    #print("-------------- Saving model --------------")

                    wandb.finish()


## Set and log hyperparams ###
# criterion = "gini"
# max_depth = 500
# max_leaf_nodes = 100
# min_samples_split = 3
# min_samples_leaf = 1
# splitter = "random"
# max_features = 300000
# class_weight = "balanced"
# algo = "DecisionTreeClassifier"
# wandb.config.update({"x_train_len" : len(X_train), "x_test_len" : len(X_test), "y_train_len": len(y_train), "y_test_len": len(y_test), "max_features" : max_features, "criterion" : criterion, "max_depth" : max_depth, "max_leaf_nodes" : max_leaf_nodes, "min_samples_split" : min_samples_split, "min_samples_leaf" : min_samples_leaf, "splitter" : splitter, "class_weight" : class_weight, "algo" : algo})
# 
# print("-------------- Extracting features --------------")
# vect = TfidfVectorizer(max_features=max_features)
# X_train_ft = vect.fit_transform(X_train)
# X_test_ft = vect.transform(X_test)
# 
# print("-------------- Scaling features --------------")
# scaler = MaxAbsScaler()
# X_train_ft = scaler.fit_transform(X_train_ft)
# X_test_ft = scaler.transform(X_test_ft)
# 
# print("-------------- Training DT --------------")
# dt = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, splitter=splitter, class_weight=class_weight, random_state=69)
# dt.fit(X_train_ft, y_train)
# 
# print("-------------- Eval DT --------------")
# y_pred = dt.predict(X_test_ft)
# 
# print("-------------- Calling W&B --------------")
# wandb.log({"classification_report": classification_report(y_test, y_pred, target_names=classes, output_dict=True)})
# 
# print("-------------- Predicting with classifier --------------")
# print(classes[dt.predict(vect.transform(["Oh my God, she's so cute!"]))[0]])
# 
print("-------------- Saving model --------------")
# 
# wandb.finish()