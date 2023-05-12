import pandas as pd
import re
import numpy as np

# Load the dataset into a pandas DataFrame
df = pd.read_csv('./datasets/sentAnal/sents_merged_cleaned_expanded.csv', sep="|")

# Define regular expressions to match URLs and Discord mentions
url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
mention_pattern = re.compile(r'<@\!?(\d+)>')

def remove_patterns(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text

df['content'] = df['content'].astype(str)

df['content'] = df['content'].apply(lambda text: remove_patterns(text, [url_pattern, mention_pattern]))

df['content'].replace('', np.nan, inplace=True)

df.dropna(subset=['content'], inplace=True)

# # Define a function to remove URLs and Discord mentions from a string
# def remove_urls_and_mentions(text):
#     if text is not None and isinstance(text, str):
#         text = re.sub(url_pattern, '', text)
#         text = re.sub(mention_pattern, '', text)
#         return text.strip()
#     else:
#         return text

# # Apply the function to the "content" column of the DataFrame
# df['content'] = df['content'].apply(remove_urls_and_mentions)

df.to_csv('./datasets/sentAnal/cleared.csv', sep='|', index=False)