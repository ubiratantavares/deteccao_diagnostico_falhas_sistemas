import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
from imblearn.over_sampling import SMOTE

'''
Foram gerados 4 cenários para o desenvolvimento da aplicação:

1- Processo operando sem falha: Falha = 0
2- Ocorrência de IDV(1): Falha = 1
3- Ocorrência de IDV(2): Falha = 2
4- Ocorrência de IDV(3): Falha = 3

'''

'''
1. carregue os bancos de dados disponibilizados
'''
ds0 = pd.read_excel('../dataset/falha_0.xlsx')
ds1 = pd.read_excel('../dataset/falha_1.xlsx')
ds2 = pd.read_excel('../dataset/falha_2.xlsx')
ds3 = pd.read_excel('../dataset/falha_3.xlsx')

'''
2. atribua uma variável "Falha" (Target) a cada banco de dados baseado no cenário de falha descrito
anteriormente.
'''
ds0['Falha'] = 0
ds1['Falha'] = 1
ds2['Falha'] = 2
ds3['Falha'] = 3

'''
3. gere informações descritivas sobre cada banco de ddos, ou seja, quantos vetores por variáveis e os tipos
'''
# verificando as informações no banco de dados
print('falha_0')
ds0.info()
print(ds0.shape)
print(ds0.describe(include='all'))

print('falha_1')
ds1.info()
print(ds1.shape)
print(ds1.describe(include='all'))

print('falha_2')
ds2.info()
print(ds2.shape)
print(ds2.describe(include='all'))

print('falha_3')
ds3.info()
print(ds3.shape)
print(ds3.describe(include='all'))


'''
4. analise se há falta de informação 'NaN'. Se houver, preencha os 'NaN' com os dados do vizinho de cima,
propagação da informação.
'''

print('falha_0')
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds0.isnull().sum())
# inserindo valores não vazios
ds0.fillna(method='ffill', inplace=True) # propagando os dados das variáveis
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds0.isnull().sum())

print(ds0.shape)
# verifica e exclui dados duplicados
ds0.drop_duplicates(inplace=True)
print(ds0.shape)

print('falha_1')
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds1.isnull().sum())
# inserindo valores não vazios
ds1.fillna(method='ffill', inplace=True) # propagando os dados das variáveis
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds1.isnull().sum())

print(ds1.shape)
# verifica e exclui dados duplicados
ds1.drop_duplicates(inplace=True)
print(ds1.shape)

print('falha_2')
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds2.isnull().sum())
# inserindo valores não vazios
ds2.fillna(method='ffill', inplace=True) # propagando os dados das variáveis
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds2.isnull().sum())

print(ds2.shape)
# verifica e exclui dados duplicados
ds2.drop_duplicates(inplace=True)
print(ds2.shape)


print('falha_3')
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds3.isnull().sum())
# inserindo valores não vazios
ds3.fillna(method='ffill', inplace=True) # propagando os dados das variáveis
# contabilizando a quantidade de informações ausentes para cada atributo
print(ds3.isnull().sum())

print(ds3.shape)
# verifica e exclui dados duplicados
ds3.drop_duplicates(inplace=True)
print(ds3.shape)


'''
5. una os bancos de dados em um dataframe. crie um arquivo único.
'''
total_linhas = ds0.shape[0] + ds1.shape[0] + ds2.shape[0] + ds3.shape[0]
print(total_linhas)
ds = pd.concat([ds0, ds1, ds2, ds3], axis=0)
print(ds.shape)

'''
6. analise a correlação entre as variáveis do problema pelo heatmap. 
'''

# verificando a matriz de correlação entre os atrbutos
corr = ds.iloc[:, 0:41].corr()
print(corr)

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

# https://seaborn.pydata.org/generated/seaborn.heatmap.html
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(12, 12))
    ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, mask=mask)

ax.set_ylim(len(corr), 0)
plt.title('Matrix de Correlação Entre os Atributos')
plt.show()

# Há redundância no comportamento? Sim

# justifique criando gráfico de barras destas variáveis "fortemente correlacionadas"

# criando um histograma de um atributo
ds.hist(column='x7', bins=10)
plt.ylabel('Frequencias')
plt.show()

ds.hist(column='x13', bins=10)
plt.ylabel('Frequencias')
plt.show()

ds.hist(column='x16', bins=10)
plt.ylabel('Frequencias')
plt.show()

'''
7. analise qual variável é mais importante através do ranking do PCA
'''
# analisando os componentes principais
X = np.asarray(ds.iloc[:, 0:41])
S = np.cov(np.transpose(X))
dig_prin = np.diagonal(S)
print(dig_prin) # valores associados da matriz covariância da diagonal principal
print("\n")

pca = PCA(n_components=41)
pca.fit(X)

print('\npca components')
pcas = pd.DataFrame(pca.components_)
print(pcas)

# ordenacao pelo critério do maior score para a primeira componente principal
pca1 = pd.DataFrame(np.round(pca.components_[0], 3))
pca1['cp'] = pca1

# ordenando de ordem decrescente
print('\n# ordenacao descrescente pelo critério do maior score para a primeira componente principal')
value_pca = pca1.sort_values(by="cp", ascending=False)
print(value_pca)

# taxa de explicação da variância
tx_var = pca.explained_variance_ratio_
print('\ntaxa de explicação da variância')
print(tx_var)

# gerar o plot
fig = plt.figure(figsize=(12, 12))
plt.plot(tx_var, 'ro-', linewidth=1)
plt.title("Scree Plot")
plt.xlabel("Principal Component Analysis")
plt.ylabel("Eigenvalue")
plt.show()

# variância acumulada
var_acum = pca.explained_variance_ratio_.cumsum()
print('\nvariância acumulada')
print(var_acum)

# entendendo pelo gráfico
x1 = np.arange(41)
width = 0.9
plt.bar(x1, var_acum, width)
plt.xlabel("CPS")
plt.ylabel("Variância Acumulada")
plt.show()

# escores das componentes principais
print('\nescores das componentes principais')
matriz_cp = pca.transform(X)
cps = pd.DataFrame(matriz_cp)
print(cps)

# refazendo a analise para somente 5 componentes principais
print('refazendo a analise para somente 5 componentes principais')
pca = PCA(n_components=5)
pca.fit(X)
print(pca.components_)
for i in range(0, 5):
    print(np.round(pca.components_[i], 3))

'''
8. analise se os dados estão desbalanceados. em caso  positivo, use método oversampling SMOTE para aumentar as classes 
minoritárias.

a. por se tratar de um problema multi-classe use o código, abaixo, identificar as porcentagens nas 
classes.
'''
X = ds[ds.columns[0:41]]
Y = ds['Falha']

counter = Counter(Y)

for k,v in counter.items():
    per = v/len(Y) * 100
    print('Class={}, n={} ({:.2f}%)'.format(k, v, per))

ax = sns.countplot(x=Y, data=ds)
np.bincount(Y)
plt.xlabel("Categoria")
plt.ylabel("Frequência")
plt.show()

'''
b- A função SMOTE tem atributos como: (*, sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None)
Para problemas multiclasses o sampling_strategy não aceita float. 
Use sampling_strategy = {Classe 1: nº de vetores desejados, Classe 2: nº de vetores desejados, etc}
'''

# aplicando SMOTE
strategy = {1:21104, 2:21104, 3:21104}
oversample = SMOTE(sampling_strategy=strategy, random_state=None, k_neighbors=5, n_jobs=None)
X, Y = oversample.fit_resample(X, Y)

counter = Counter(Y)

for k,v in counter.items():
    per = v/len(Y) * 100
    print('Class={}, n={} ({:.2f}%)'.format(k, v, per))

ax = sns.countplot(x=Y, data=ds)
np.bincount(Y)
plt.xlabel("Categoria")
plt.ylabel("Frequência")
plt.show()