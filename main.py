import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import warnings
import sys
import os
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
from sklearn.cluster import cluster_optics_dbscan
from matplotlib import gridspec as gridspec

data = pd.read_csv('CC GENERAL.csv').iloc[:, 1:].dropna()
# KMeans Clustering
# Step 1
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=15).fit(data)

# Step 2
data = np.array(data, dtype='float')
scaler = preprocessing.StandardScaler()
scaled_data = scaler.fit_transform(data)

# Step 3
clustering = DBSCAN().fit(scaled_data)
print(set(clustering.labels_))
print(len(set(clustering.labels_)) - 1)
print(list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

# Step 4
max_eps = 0
eps_list = []
count_clusters = []
percent_not_clusters_labels = []
while max_eps <= 2:
    max_eps += 0.1
    eps_list.append(max_eps)
    clustering = DBSCAN(eps=max_eps).fit(scaled_data)
    count_clusters.append(len(set(clustering.labels_)) - 1)
    percent_not_clusters_labels.append((len(set(clustering.labels_)) - 1) * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

fig, ax = plt.subplots()
ax.bar(eps_list, count_clusters, width=0.05, color='mediumorchid', label='Number of Clusters')
ax.bar(eps_list, percent_not_clusters_labels, width=0.05, color='slateblue', label='% of Noise Points')
ax.set_facecolor('seashell')
fig.set_figwidth(16)
fig.set_figheight(8)
fig.set_facecolor('floralwhite')
plt.xlabel('Maximum Distance')
plt.ylabel('Number of Clusters')
plt.legend()
plt.show()

# Step 5
min_samples = 1
samples_min_list = []
count_clusters = []
percent_not_clusters_labels = []
while min_samples <= 12:
    min_samples += 1
    samples_min_list.append(min_samples)
    clustering = DBSCAN(min_samples=min_samples).fit(scaled_data)
    count_clusters.append(len(set(clustering.labels_)) - 1)
    percent_not_clusters_labels.append((len(set(clustering.labels_)) - 1) * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

fig, ax = plt.subplots()
ax.bar(samples_min_list, count_clusters, color='mediumslateblue', label='Number of Clusters')
ax.bar(samples_min_list, percent_not_clusters_labels, color='steelblue', label='% of Noise Points')
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(8)
fig.set_facecolor('floralwhite')
plt.xlabel('Minimum Number of Samples')
plt.ylabel('Number of Clusters')
plt.legend()
plt.show()

# Step 6
clustering = DBSCAN(eps=2, min_samples=3).fit(scaled_data)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

# Step 7
pca = PCA(n_components=2).fit(scaled_data)
clustering = DBSCAN(eps=2, min_samples=3).fit(scaled_data)
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
colors = [plt.cm.get_cmap("Spectral")(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
    class_member_mask = labels == k
    xy = scaled_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=14)
    xy = scaled_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], "o", markerfacecolor=tuple(col), markeredgecolor="k", markersize=6)
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

clustering = OPTICS(max_eps=2, min_samples=3, cluster_method='dbscan').fit(scaled_data)
print(f'With max_eps=2 and min_samples=3 the number of clusters is: {len(set(clustering.labels_)) - 1} and the percentage of unclustered data is: {list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100}%')

#3
labels_050 = cluster_optics_dbscan(
    reachability=clustering.reachability_,
    core_distances=clustering.core_distances_,
    ordering=clustering.ordering_,
    eps=0.5,
)
labels_200 = cluster_optics_dbscan(
    reachability=clustering.reachability_,
    core_distances=clustering.core_distances_,
    ordering=clustering.ordering_,
    eps=2,
)
space = np.arange(len(scaled_data))
reachability = clustering.reachability_[clustering.ordering_]
labels = clustering.labels_[clustering.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])
colors = ["b.", "g.", "r.", "c.", "y."]
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)
ax1.plot(space, np.full_like(space, 2.0, dtype=float), "k-", alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), "k-.", alpha=0.5)
ax1.set_ylabel("Reachability (epsilon distance)")
ax1.set_title("Reachability Plot")
colors = ["b.", "g.", "r.", "c.", "y."]
for klass, color in zip(range(0, 5), colors):
    Xk = scaled_data[clustering.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(scaled_data[clustering.labels_ == -1, 0], scaled_data[clustering.labels_ == -1, 1], "k+", alpha=0.1)
ax2.set_title("Automatic Clustering\nOPTICS")

colors = ["b.", "g.", "r.", "c."]
for klass, color in zip(range(0, 4), colors):
    Xk = scaled_data[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax3.plot(scaled_data[labels_050 == -1, 0], scaled_data[labels_050 == -1, 1], "k+", alpha=0.1)
ax3.set_title("Clustering at 0.5 epsilon cut\nDBSCAN")

colors = ["b.", "m.", "r.", "c."]
for klass, color in zip(range(0, 4), colors):
    Xk = scaled_data[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(scaled_data[labels_200 == -1, 0], scaled_data[labels_200 == -1, 1], "k+", alpha=0.1)
ax4.set_title("Clustering at 2.0 epsilon cut\nDBSCAN")
plt.tight_layout()
plt.show()


#4
optic_metrics = ('cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan')
for metric in optic_metrics:
    clustering = OPTICS(max_eps=2, min_samples=3, metric=metric).fit(scaled_data)
    print(f'With max_eps=2, min_samples=3, and metric={metric} the number of clusters is: {len(set(clustering.labels_)) - 1} and the percentage of unclustered data is: {list(clustering.labels_).count(-1) / len(list(clustering.labels_)) * 100}%')