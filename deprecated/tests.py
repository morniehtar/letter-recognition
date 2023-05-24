import time
import numpy as np
f = open("./logs/output.log", 'w')

imgs = np.load("imgs_tr.npy")
lbls = np.load("lbls_tr.npy")
alphbt = np.unique(lbls)
print(time.process_time(), "array init complete", file=f)

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=100, whiten=True,
          svd_solver='randomized', random_state=615)
svc = SVC(kernel='rbf', class_weight='balanced', C=50, gamma='auto')
model = make_pipeline(pca, svc)
print(time.process_time(), "model init complete", file=f)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(imgs, lbls, random_state=615)

print(time.process_time(), "data split complete", file=f)

model.fit(Xtrain, ytrain)
print(time.process_time(), "fit complete", file=f)

yfit = model.predict(Xtest)
print(time.process_time(), "predict complete", file=f)

from sklearn.metrics import classification_report
print(classification_report(np.array(ytest), yfit, target_names=alphbt), file=f)

import matplotlib; matplotlib.use('QtCairo')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(ytest, yfit)
plt.figure(dpi=160, figsize=(9, 6))
sns.heatmap(mat.T, annot=True, fmt='d',
            cbar=False, cmap='YlOrBr',
            xticklabels=alphbt,
            yticklabels=alphbt)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.tight_layout()
plt.show()
