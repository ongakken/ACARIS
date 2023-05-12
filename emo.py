###########
## Simtoon, 2023
## Ongakken s. r. o.
## Handle with care
###########

from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline, logging
import tensorflow as tf
import numpy as np
import os
import csv
from alert import send_alert

class emo:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # kill off logging
        logging.set_verbosity_error() # kill off logging
        self.tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa") # set the tokenizer from our repo
        self.model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa") # set the model and pull from our repo, if missing on local
        self.emotion = pipeline("sentiment-analysis", model="arpanghoshal/EmoRoBERTa") # make the initial prediciton using our model
        self.label2id = {
            "admiration": 0, # ahhh, what a clean code!
            "amusement": 1, # amusement?? Yeah, I feel that each time I see a Python developer
            "anger": 2, # this is what I feel when I have to fuck around with Python's whitespace sensitive syntax, ffs
            "annoyance": 3, # my genuine feeling when I have to deal with Python's GIL
            "approval": 4, # also known as "Holy fuck, that ran with 0 warnings!!"
            "caring": 5, # I only feel this when coding in C++. Sorry, Python
            "confusion": 6, # yeah, each time I have to look at somebody else's Python code
            "curiosity": 7, # when somebody tells me they are working on a cool new RL project
            "desire": 8, # when you see a clean, well-structured cpp class. almost like seeing a cute girl walking down the street
            "disappointment": 9, # when Python simply isn't cutting it
            "disapproval": 10, # when seeing somebody choosing Python over C++ for no apparent reason
            "disgust": 11, # bleh, that Python's built-in string formatting
            "embarrassment": 12, # when you accidentally leak your prompt and people see what sick things you told that poor LLM to do
            "excitement": 13, # when you realize that you're free to use C++ instead of Python
            "fear": 14, # yeah, this is how I feel when pushing to prod without testing beyond on my own machine
            "gratitude": 15, # the feeling of accomplishment when somebody says your code is good
            "grief": 16, # that crippling feeling when you see Python as #1 lang
            "joy": 17, # when my cpp code finally compiles after dealing with that sneaky bastard of a bug for weeks on end
            "love": 18, # the strong feeling when my code performs better than my last girlfriend
            "nervousness": 19, # sure, I feel this every day while the compiler is running
            "optimism": 20, # when you see that IntelliCode reports 200 errors, but you're optimistic that the compiler won't mind
            "pride": 21, # when you see that cpp just runs faster than Python
            "realization": 22, # when you're thankful that you picked cpp in the early days instead of Python
            "relief": 23, # when you finally close up that non-optional Python project and finally return to cpp
            "remorse": 24, # the feeling when looking at this very code and knowing that it's written in Python
            "sadness": 25, # you feel this when seeing that Python is only getting more and more popular
            "surprise": 26, # "wow, that run on the first try?"
            "neutral": 27 # the sentiment of my friends every time I ask how their day is going
        }

    def predict(self, txt): # this method handles the prediction. if only I could comment in a header file next to the declaration ...
        try: # error handling
            emotionLabels = self.emotion(txt) # this runs the prediction itself
        except Exception as e: # if something goes wrong
            print(f"Err: {str(e)}")
            return "Err" # in case of an error, the func will return "Err" instead of a sentiment

        if emotionLabels[0]["score"] > 0.75: # if our confidence is higher than .75
            predictedEmotion = emotionLabels[0]["label"] # we pull the label and store it in str predictedEmotion
            predictedEmotionInt = self.label2id[predictedEmotion.lower()] # we store the int representation of the predictedEmotion in the predictedEmotionInt var, based on the dict in the beginning
            if emotionLabels[0]["score"] < 0.95: # if our confidence is lower than .95
                print("current prediction: ", predictedEmotion) # we print the prediction
                print("current confidence: ", emotionLabels[0]["score"]) # we print the confidence
                print("input: ", txt) # we print the input
                # while True: # loop until broken
                #     correctLabel = input("Correct label: ") # ask for correction of our predicted label
                #     correctLabelInt = self.label2id.get(correctLabel.lower()) # based on the correction entered, look up the int value for the label
                #     if correctLabelInt is None: # if not found in the dict, prompt for correction again 
                #         print("Invalid label! Retry!!\n")
                #     else:
                #         break # break when the entered emotion matches a label from the dict
                
                # if predictedEmotionInt != correctLabelInt: # if the entered emotion is different that the one we predicted ...
                #     self.updateModel(txt, correctLabelInt) # ... retrain the model to include this seq and the correct label. over time, this will improve accuracy, but it's gonna take a long time

        else: # if our confidence is lower or equal to .75 ...
            # return f"Not sure ... rather not continuing!\n----------------\nDebug:\n----------------\nclass: {emotionLabels[0]['label']}\nconfidence: {emotionLabels[0]['score']}\n" # ... we cannot trust the prediction in this high-stakes circumstance
            predictedEmotion = emotionLabels[0]["label"]
            return predictedEmotion # not trustworthy, but for this use-case, better than NaN
        
        return predictedEmotion # if all goes well, we return the prediction here

    def updateModel(self, input, correctLabelInt): # retrain using the input + corrected label
        x = self.tokenizer(input, return_tensors="tf") # tokenize the input seq
        y = np.array(correctLabelInt) # set the label as the sole member of this list (batch)
        print(y)
        self.model.compile(optimizer="adam") # compile the model
        self.model.fit(x, y) # run training on the sole-member batch

emo_mdl = emo() # instantiate the class
with open("./datasets/msgsNew.csv", "r") as f: # open the input file
    reader = csv.DictReader(f, delimiter="|", fieldnames=["uid", "timestamp", "content"])

    with open("./datasets/msgsNewLabeled.csv", "w", newline="") as out:
        fieldnames = ["uid", "timestamp", "content", "sentiment"]
        writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()

        for idx, row in enumerate(reader):
            if not all(key in row for key in ["uid", "timestamp", "content"]):
                raise ValueError(f"Row {idx} is missing a key: {row}")
            print(row)
            msg = row["content"]
            inferredSentiment = emo_mdl.predict(msg)
            newRow = {"uid": row["uid"], "timestamp": row["timestamp"], "content": row["content"], "sentiment": inferredSentiment}
            writer.writerow(newRow)
        send_alert("AAAAAAH! I FINISHED!!!", "Emotion analysis finished!", "normal", 5000)
# print(emo_mdl.predict("Oh my God, she's so cute!!!")) # this quoted part is the actual input seq, so we print the return of the predict() method, passing the seq to it