import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Step 1: Load pickled feature vectors for each user
user_feats = {}
for file in os.listdir("./pickles"):
	if file.endswith("_feats.pkl"):
		uid = file.split("_")[0] + "#" + file.split("_")[1].replace(".pkl", "")
		with open(os.path.join("./pickles", file), "rb") as f:
			feats = pickle.load(f)
		user_feats[uid] = feats

for uid, feats in user_feats.items():
	if not feats:
		clnFeats = {uid: feats for uid, feats in user_feats.items() if feats}

# Step 2: Use PCA to reduce the dimensionality of the feature vectors
pca = PCA(n_components=50)
user_feats_reduced = {}
for uid, feats in clnFeats.items():
	if not feats or len(feats) == 0:
		print(f"User {uid} has no features")
		continue
	print(uid, type(feats))
	feats = [feat for feat in feats if isinstance(feat, dict)]
	feats = np.array([list(feat.values()) for feat in feats])
	feats_reduced = pca.fit_transform(feats)
	user_feats_reduced[uid] = feats_reduced

# Step 3: Combine the reduced feature vectors for all users into a single matrix
user_feats_combined = np.concatenate(list(user_feats_reduced.values()), axis=0)

# Step 4: Normalize the combined feature matrix
scaler = StandardScaler()
user_feats_combined_scaled = scaler.fit_transform(user_feats_combined)

# Step 5: Use t-SNE to embed the users into a 2D space
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
user_embeddings = tsne.fit_transform(user_feats_combined_scaled)

# Step 6: Plot the embedded users using a scatter plot
plt.figure(figsize=(10,10))
for i, uid in enumerate(user_feats_reduced.keys()):
	plt.scatter(user_embeddings[i,0], user_embeddings[i,1], label=uid)
plt.legend()
plt.show()