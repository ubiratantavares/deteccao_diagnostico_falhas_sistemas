import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')
print(dataset.columns)

# verificando a quantidade de dados em cada classe
count_classes = pd.value_counts(dataset['Y'], sort=True).sort_index()
print(count_classes)

X = dataset[dataset.columns[0:9]]
Y = dataset['Y']

# visualizando os dados originais
Y.value_counts()
ax = sns.countplot(x=Y, data=dataset)
np.bincount(Y)
plt.xlabel("Categoria")
plt.ylabel("Frequência")
plt.show()













