import numpy as np
imgs = np.load("imgs.npy")
lbls = np.load("lbls.npy")

uni, cnt = np.unique(lbls, return_counts=True)
print(dict(zip(uni, cnt)))
cs = np.cumsum(cnt)

alphb = np.empty(26, dtype=np.str_)
idx = np.concatenate(  (np.arange(0+5000, 0+6000), \
                        np.arange(cs[0]+5000, cs[0]+6000), \
                        np.arange(cs[1]+5000, cs[1]+6000), \
                        np.arange(cs[2]+5000, cs[2]+6000), \
                        np.arange(cs[3]+5000, cs[3]+6000), \
                        np.arange(cs[4], cs[4]+900), \
                        np.arange(cs[5]+5000, cs[5]+6000), \
                        np.arange(cs[6]+5000, cs[6]+6000), \
                        np.arange(cs[7], cs[7]+841), \
                        np.arange(cs[8]+5000, cs[8]+6000), \
                        np.arange(cs[9]+4000, cs[9]+5000), \
                        np.arange(cs[10]+5000, cs[10]+6000), \
                        np.arange(cs[11]+5000, cs[11]+6000), \
                        np.arange(cs[12]+5000, cs[12]+6000), \
                        np.arange(cs[13]+5000, cs[13]+6000), \
                        np.arange(cs[14]+5000, cs[14]+6000), \
                        np.arange(cs[15]+4000, cs[15]+5000), \
                        np.arange(cs[16]+5000, cs[16]+6000), \
                        np.arange(cs[17]+5000, cs[17]+6000), \
                        np.arange(cs[18]+5000, cs[18]+6000), \
                        np.arange(cs[19]+5000, cs[19]+6000), \
                        np.arange(cs[20]+2000, cs[20]+3000), \
                        np.arange(cs[21]+5000, cs[21]+6000), \
                        np.arange(cs[22]+5000, cs[22]+6000), \
                        np.arange(cs[23]+5000, cs[23]+6000), \
                        np.arange(cs[24]+4000, cs[24]+5000)) )

imgs = imgs[idx]
lbls = lbls[idx]

#print(lbls[2999])

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