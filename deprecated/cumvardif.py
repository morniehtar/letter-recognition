import numpy as np

imgs = np.load("imgs.npy")
lbls = np.load("lbls.npy")

import matplotlib as mpl; mpl.use('QtCairo')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA().fit(imgs)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.grid()
plt.tight_layout()
plt.show()