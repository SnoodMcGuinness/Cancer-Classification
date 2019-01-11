#Imports the data, normalises it and splits into 
#training and testing sets

import pandas as pd

def import_data(filename):
    #Generate column headers
    features=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave points', 'Symmetry', 'Fractal dimension']

    col_names=['Classification']
    for feature in features:
        col_names.append(feature + ' mean')
        col_names.append(feature + ' std err')
        col_names.append(feature + ' largest value')
    
    #Read data
    dataset=pd.read_csv(filename, header=None, names=col_names, index_col=0)
    
    #create labels
    labels=dataset['Classification']=='M'
    labels=labels.values
    
    #normalise training data
    features=dataset.drop('Classification',axis=1).values
    return [dataset, labels, features]
    