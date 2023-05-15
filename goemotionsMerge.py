import csv

dataset = []
with open('./datasets/go_emotions_dataset.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        dataset.append(row)

def merge_labels(row):
    emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
                'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']

    pos_emotions = ['admiration', 'amusement', 'approval', 'caring', 'curiosity', 'excitement',
                    'gratitude', 'joy', 'love', 'optimism', 'pride', 'realization', 'relief', 'surprise']

    neg_emotions = ['anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
                    'fear', 'grief', 'remorse', 'sadness']

    pos_count = sum(int(row[emotion]) for emotion in pos_emotions)
    neg_count = sum(int(row[emotion]) for emotion in neg_emotions)
    neu_count = int(row['neutral'])

    if pos_count > neg_count and pos_count > neu_count:
        row['sentiment'] = 'pos'
    elif neg_count > pos_count and neg_count > neu_count:
        row['sentiment'] = 'neg'
    else:
        row['sentiment'] = 'neu'

    for emotion in emotions:
        del row[emotion]

    del row['example_very_unclear']

    return row

for i, row in enumerate(dataset):
    dataset[i] = merge_labels(row)

fieldnames = dataset[0].keys()

with open('./datasets/go_merged.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='|')
    writer.writeheader()
    writer.writerows(dataset)