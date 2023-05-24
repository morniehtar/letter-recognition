import time
import numpy as np
f = open("./logs/output.log", 'w')

imgs = np.load("imgs_tr.npy")
lbls = np.load("lbls_tr.npy")
alphbt = np.unique(lbls)
print(time.process_time(), "array init complete")

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=100, whiten=True,
          svd_solver='randomized', random_state=615)
svc = SVC(kernel='rbf', C=1e9, gamma=1)
model = make_pipeline(pca, svc)
#model = SVC(kernel='rbf', C=1e6, gamma=0.01)
print(time.process_time(), "model init complete")

proc1 = time.process_time()
model.fit(imgs, lbls)
proc2 = time.process_time()
print( time.process_time(), "fit complete in %f sec"%(proc2-proc1) )

import pickle
pickle.dump(model, open('model.sav', 'wb'))