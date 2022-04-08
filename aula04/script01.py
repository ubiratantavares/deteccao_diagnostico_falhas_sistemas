import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# visualizando o banco de dados
print(dataset.head())

# verificando as informações no banco de dados
dataset.info()

print(dataset.describe(include='all'))

# analisando ausência de informações
print(dataset.isnull()) # retorna True para todos os elementos que não possuem informações

# contabilizando a quantidade de informações ausentes para cada atributo
print(dataset.isnull().sum())

# verificando a matriz de correlação entre os atrbutos
corr = dataset.iloc[:, 0:9].corr()
print(corr)

# exibindo o mapa de calor da matriz de correlação
fig = plt.figure(figsize=(20, 10))

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
ax = sns.heatmap(corr, vmin = -1, vmax = 1, center = 0, square = True, annot = True, linewidths = 0.8)
ax.set_ylim(len(corr), 0)
plt.title('Matrix de Correlação Entre os Atributos')
plt.show()
