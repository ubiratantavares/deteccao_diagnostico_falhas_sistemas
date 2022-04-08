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
values = {"x1": 0.1, "x2": 0.5, "x3": 0.8}
dframe = dataset.fillna(value=values)
print(dframe)
with pd.ExcelWriter("../dataset/dados_cancer_ex1_v_esp_v.xlsx") as writer:
  dframe.to_excel(writer)

print(dframe.isnull().sum())
