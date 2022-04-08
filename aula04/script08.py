import pandas as pd
import numpy as np

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# analisando os componentes principais
X = np.asarray(dataset.iloc[:, 0:9])

pca = PCA(n_components=9)

pca.fit(X)

pcas = pd.DataFrame(pca.components_)
print(pcas)

# ordenacao pelo critério do maior score para a primeira componente principal
pca1 = pd.DataFrame(np.round(pca.components_[0], 3))
pca1['cp'] = pca1
print(pca1)

# ordenando de ordem decrescente
value_pca = pca1.sort_values(by="cp", ascending=False)
print(value_pca)

