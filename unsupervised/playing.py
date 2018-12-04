from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))




data = pd.read_pickle('data_gr_100.pkl')

sample_size = 100
n_samples, n_features = data.shape
n_classes = len(np.unique(data.label))
labels = data.label




pca = PCA(n_components=n_classes).fit(data)
#bench_k_means(KMeans(init=pca.components_, n_clusters=n_classes, n_init=1),
#              name="PCA-based",
#              data=data)
#print(82 * '_')



# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=15).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_classes, n_init=10)
kmeans.fit(reduced_data)
