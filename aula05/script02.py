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

X = dataset[dataset.columns[0:9]]
Y = dataset['Y']

# verificando a quantidade de dados em cada classe
count_classes = pd.value_counts(Y, sort=True).sort_index()
print(count_classes)

perc = np.array(count_classes)[0]/np.array(count_classes)[1]
print("{:.1f}%".format((perc * 100.0)))

# aplicando SMOTE
X, Y = SMOTE(sampling_strategy=0.9).fit_resample(X, Y)

# verificando a quantidade de dados em cada classe
count_classes = pd.value_counts(Y, sort=True).sort_index()
print(count_classes)

perc = np.array(count_classes)[0]/np.array(count_classes)[1]
print("{:.1f}%".format((perc * 100.0)))

# juntando os datafranes e salvando no excel
data_balance = pd.concat([X, Y], axis=1)
writer = pd.ExcelWriter('../dataset/dados_balanceados.xlsx')
data_balance.to_excel(writer, 'plan1')
writer.save()

# viualizando os dados balanceados
ax = sns.countplot(x=Y, data=data_balance)
np.bincount(Y)
plt.xlabel("Categoria")
plt.ylabel("Frequência")
plt.show()
