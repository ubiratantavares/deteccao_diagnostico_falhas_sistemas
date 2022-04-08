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

# excluindo linhas com ausência de informações
n_linhas_t = dataset.shape[0]
print(n_linhas_t) # verificando o número de linhas do dataframe

n_linhas_at = dataset.dropna(subset=["x1"]).shape[0] # retirada dos dados nulos da coluna x1
print(n_linhas_at) # verificando o número de linhas do dataframe apos exclusão das linhas que são NaN

linhas_exec = n_linhas_t - n_linhas_at
print(linhas_exec) # verificando o número de linhas uteis do dataframe

por_exc = (1 - (n_linhas_at/n_linhas_t)) * 100
print('Porcentagem de linhas excluídas = {:.1f}%'.format(por_exc)) # verifica o percentual de linhas excluidas

dframe = dataset.dropna() # criando um dataframe com os dados sem NaN
print(dframe) # retorna o dataframe sem NaN

with pd.ExcelWriter("../dataset/dados_cancer_ex1_s_nulos_x1.xlsx") as writer:
  dframe.to_excel(writer)