import pandas as pd
import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# analisando os componentes principais
X = np.asarray(dataset.iloc[:, 0:9])

pca = PCA(n_components=5) # refazendo a analise para somente 5 componentes principais

pca.fit(X)

print(pca.components_)

for i in range(0, 5):
    print(np.round(pca.components_[i], 3))

