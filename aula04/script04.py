import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# analisando os componentes principais
X = np.asarray(dataset.iloc[:, 0:9])

S = np.cov(np.transpose(X))

dig_prin = np.diagonal(S)
print(dig_prin) # valores associados da matriz covariância da diagonal principal
print("\n")

pca = PCA(n_components=9)
p1 = pca.fit(X)
print(p1)
print("\n")

p2 = pca.components_[0:9, :]
print(p2)
print("\n")

tx_var = pca.explained_variance_ratio_ # taxa de explicação da variância
print(tx_var)
print("\n")

# gerar o plot
fig = plt.figure(figsize=(9,5))
plt.plot(tx_var, 'ro-', linewidth=2)
plt.title("Scree Plot")
plt.xlabel("Principal Component Analysis")
plt.ylabel("Eigenvalue")
plt.show()