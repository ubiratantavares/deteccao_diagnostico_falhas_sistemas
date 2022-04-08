import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# verificando a matriz de correlação entre os atrbutos
corr = dataset.iloc[:, 0:9].corr()
print(corr)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# exibindo o mapa de calor da matriz de correlação
fig = plt.figure(figsize=(20, 10))

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, annot=True, linewidths=0.8, mask=mask)

ax.set_ylim(len(corr), 0)
plt.title('Matrix de Correlação Entre os Atributos')
plt.show()
