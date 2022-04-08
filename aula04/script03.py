import pandas as pd
import matplotlib.pyplot as plt

# inserindo o banco de dados
dataset = pd.read_excel('../dataset/data_cancer.xlsx')

# criando um histograma de um atributo
dataset.hist(column='x1', bins=10)
plt.ylabel('Frequencias')
plt.show()

dataset.hist(column='x2', bins=20)
plt.ylabel('Frequencias')
plt.show()

dataset.hist(column='x3', bins=20)
plt.ylabel('Frequencias')
plt.show()
