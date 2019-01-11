#Visualise the dataset
#Based heavily on https://www.kaggle.com/benhamner/python-data-visualizations

from data_import import import_data
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style='white', color_codes=True)

dataset, labels, features = import_data('Exercise1 - data.csv')

print(dataset['Classification'].value_counts())
plot=sns.pairplot(dataset, hue='Classification')
