import pandas as pd
import numpy as np

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/dados_cancer_ex1.xlsx')

# verificando as informações no banco de dados
dataset.info()

print(dataset.describe(include='all'))

# analisando ausência de informações
print(dataset.isnull()) # retorna True para todos os elementos que não possuem informações

# contabilizando a quantidade de informações ausentes para cada atributo
print(dataset.isnull().sum())

# inserindo valores não vazios
dframe = dataset.fillna(method='ffill') # propagando os dados das variáveis

n_linhas_t = dframe.shape[0]

print(dframe.isnull().sum())

dframe = dframe.drop_duplicates()

n_linhas_at = dframe.drop_duplicates().shape[0]

linhas_exc = n_linhas_t - n_linhas_at
print(linhas_exc)

por_exc = (1 - (n_linhas_at/n_linhas_t)) * 100

print('Porcentagem excluida do banco de dados foi {:.1f}%'.format(por_exc))

with pd.ExcelWriter("../dataset/dados_cancer_ex1_a_desn2.xlsx") as writer:
  dframe.to_excel(writer)

print(dframe.isnull().sum())