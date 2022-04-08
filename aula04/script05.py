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

p2 = pca.components_[0:9, :]
print(p2)
print("\n")

var_acum = pca.explained_variance_ratio_.cumsum() # variância acumulada
print(var_acum)
print("\n")

# salvar os valores da variância acumulada
df = pd.DataFrame(var_acum)
writer = pd.ExcelWriter("../dataset/variancia.xlsx")
df.to_excel(writer)
writer.save()

# entendendo pelo gráfico
x1 = np.arange(9)
width = 0.9
plt.bar(x1, var_acum, width)
plt.xlabel("CPS")
plt.ylabel("Variância Acumulada")
plt.show()









