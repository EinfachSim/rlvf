from proj.data import ValueProfileSampler
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

test = ValueProfileSampler()
profiles = test.sample_batch(100)
# 5. MDS to recover the structure
# We run MDS on the VALUES (the columns) to see if they form a circle
mds = MDS(n_components=2, metric='euclidean', random_state=42, normalized_stress='auto')
coords = mds.fit_transform(profiles.T)

# Plotting
plt.figure(figsize=(10, 10))
plt.scatter(coords[:, 0], coords[:, 1], s=100, c=np.arange(len(test.value_names)), cmap='hsv')

for i, txt in enumerate(test.value_names):
    plt.annotate(txt, (coords[i, 0], coords[i, 1]), xytext=(8, 8), textcoords='offset points', fontsize=9)

plt.title("MDS of 19 Refined Schwartz Values (Simulated Profiles)")
plt.axhline(0, color='black', alpha=0.1)
plt.axvline(0, color='black', alpha=0.1)
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()