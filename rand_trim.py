import numpy as np
from random import shuffle
imgs = np.load("imgs.npy")
lbls = np.load("lbls.npy")

uni, cnt = np.unique(lbls, return_counts=True)

print(dict(zip(uni, cnt)))
cs = np.cumsum(cnt)
print(cs)

idx = []
tmp = np.arange(0, cs[0])
shuffle(tmp)
idx= idx + list(tmp)[:1000]
for i in range(25):
    tmp = np.arange(cs[i], cs[i + 1])
    if i!=5 or i!=8:
        shuffle(tmp)
        idx = idx + list(tmp)[:1000]
    else:
        idx= idx + list(tmp)

imgs = imgs[idx]
lbls = lbls[idx]

import os
try:
    os.remove('imgs_tr.npy')
except OSError:
    pass
try:
    os.remove('lbls_tr.npy')
except OSError:
    pass

np.save('imgs_tr', imgs)
np.save('lbls_tr', lbls)
