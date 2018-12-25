import numpy as np
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# 文本处理，构建特征值
text = []
labels = []
file = open('../../dataset/Tweets.txt', 'r')
for item in file:
    # print(item)
    item = json.loads(item)
    text.append(item['text'])
    labels.append(int(item['cluster']))

labels = np.asarray(labels)
classes = np.unique(labels).shape[0]
text_vector = TfidfVectorizer(max_features=1000, use_idf=True, stop_words='english')
X = text_vector.fit_transform(text)
X = X.toarray()
# print(X)

# KMeans
km = KMeans(n_clusters=classes)
km.fit(X)
y = km.labels_
print('The NMI of KMeans is:' + str(normalized_mutual_info_score(labels, y)))

# AffinityPropagation
affinity_propagation = AffinityPropagation(damping=0.9, preference=-1)
affinity_propagation.fit(X)
y = affinity_propagation.labels_
print('The NMI of AffinityPropagation is:' + str(normalized_mutual_info_score(labels, y)))

# Mean-shift
mean_shift = MeanShift(bandwidth=0.8, bin_seeding=True)
mean_shift.fit(X)
y = mean_shift.labels_
print('The NMI of Mean-shift is:' + str(normalized_mutual_info_score(labels, y)))

# Spectral clustering
spectral = SpectralClustering(n_clusters=classes)
spectral.fit(X)
y = spectral.labels_
print('The NMI of Spectral Clustering is:' + str(normalized_mutual_info_score(labels, y)))

# Ward hierarchical clustering
connectivity = kneighbors_graph(X, n_neighbors=200, include_self=False)
connectivity = 0.5 * (connectivity + connectivity.T)
ward = AgglomerativeClustering(n_clusters=classes, linkage='ward', connectivity=connectivity)
ward.fit(X)
y = ward.labels_
print('The NMI of Ward hierarchical clustering is:' + str(normalized_mutual_info_score(labels, y)))

# Agglomerative clustering
agglomerative = AgglomerativeClustering(n_clusters=classes, linkage='average', connectivity=connectivity)
agglomerative.fit(X)
y = agglomerative.labels_
print('The NMI of Agglomerative clustering (average) is:' + str(normalized_mutual_info_score(labels, y)))

agglomerative = AgglomerativeClustering(n_clusters=classes, linkage='complete', connectivity=connectivity)
agglomerative.fit(X)
y = agglomerative.labels_
print('The NMI of Agglomerative clustering (complete) is:' + str(normalized_mutual_info_score(labels, y)))

# DBSCAN
dbscan = DBSCAN(eps=0.9, min_samples=1, algorithm='auto')
dbscan.fit(X)
y = dbscan.labels_
print('The NMI of DBSCAN NMI is:' + str(normalized_mutual_info_score(labels, y)))

# Gaussian mixtures
gm = GaussianMixture(n_components=classes)
gm.fit(X)
y = gm.predict(X)
print('The NMI of Gaussian Mixture NMI is:' + str(normalized_mutual_info_score(labels, y)))
