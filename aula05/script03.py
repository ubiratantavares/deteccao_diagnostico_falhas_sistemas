import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA

# inserindo o banco de dados
dataset1 = pd.read_excel('../dataset/data_cancer.xlsx')
x = pd.DataFrame(dataset1)

dataset2 = pd.read_excel('../dataset/data_cancer1.xlsx')
x1= pd.DataFrame(dataset2)
















