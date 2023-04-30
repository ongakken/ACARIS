import pickle

with open("pickles/NeutralH+_feats.pkl", "rb") as f:
	feats = pickle.load(f)

print(feats)