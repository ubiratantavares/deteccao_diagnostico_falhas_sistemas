import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# analisando os componentes principais
X = np.asarray(dataset.iloc[:, 0:9])

pca = PCA(n_components=9)

pca.fit(X)

# salvar  matriz CP
matriz_cp = pca.transform(X) # escores das componentes principais
cps = pd.DataFrame(matriz_cp)
print(cps)
with pd.ExcelWriter("../dataset/cps.xlsx") as writer:
    cps.to_excel(writer)
