import time
import numpy as np

imgs = np.load("imgs_tr.npy")
lbls = np.load("lbls_tr.npy")
alphbt = np.unique(lbls)
print(time.process_time(), "array init complete")

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(imgs, lbls)

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pca = PCA(n_components=100, whiten=True,
          svd_solver='randomized', random_state=615)
svc = SVC(kernel='rbf', C=50, gamma=.01, class_weight='balanced')
model = make_pipeline(pca, svc)
print(time.process_time(), "model init complete")

proc1 = time.process_time()
model.fit(Xtrain, ytrain)
proc2 = time.process_time()
print( time.process_time(), "fit complete in %f sec"%(proc2-proc1) )

ypred = model.predict(Xtest)
from sklearn.metrics import classification_report
print(classification_report(ypred, ytest))

targ = np.empty((6760, 28*28), dtype=np.int64)

import os
from PIL import Image
path = "./data/test"
iter = sorted(os.listdir(path))
for fn in iter:
    with Image.open(os.path.join(path, fn)) as img:
        targ[int(fn.replace('.png', '')), :] = np.array(img.getdata())
print(time.process_time(), "target loaded")

proc1 = time.process_time()
prd = model.predict(targ)
proc2 = time.process_time()
print( time.process_time(), "predict complete in %f sec"%(proc2-proc1) )

import json
with open("test.json", "w") as outfile:
    json.dump(dict(zip(range(6760), prd)), outfile, indent=2)