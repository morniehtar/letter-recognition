import os
from PIL import Image
import numpy as np

path = "./data/train"
imgs = np.ndarray((364244, 28*28), dtype=np.int64)
lbls = np.empty(364244, dtype=np.str_)

iter = sorted(os.listdir(path))
k = 0
for fn in iter:
    with Image.open(os.path.join(path, fn)) as img:
        width, height = img.size
        chunks_hor = width//28
        chunks_vert = height//28
        for i in range(chunks_hor):
            for j in range(chunks_vert):
                left = i * 28
                upper = j * 28
                right = ( i+1 ) * 28
                lower = ( j+1 ) * 28
                chunk = img.crop((left, upper, right, lower))
                imgs[k, :] = np.array(chunk.convert('L').getdata())
                lbls[k] = fn.replace('.png', '')
                k += 1
    print(iter.index(fn))

try:
    os.remove('imgs.npy')
except OSError:
    pass
try:
    os.remove('lbls.npy')
except OSError:
    pass

np.save('imgs', imgs)
np.save('lbls', lbls)