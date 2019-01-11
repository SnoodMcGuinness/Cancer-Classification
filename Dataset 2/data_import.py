from PIL import Image
import numpy as np
import pandas as pd
from glob import glob

def import_data(labels_file, image_folder):
    
    #Read the labels
    dataset=pd.read_csv(labels_file, index_col=0)

    #Import images
    images={}
    for file in glob(image_folder + '/*.gif'):
        images[str(file[14:-4])]=np.array(Image.open(str(file)))
        
    #Create complete dataset with matched IDs
    dataset['Image'] = pd.Series(images)
    
    #Output labels and images in numpy arrays correctly
    labels=dataset['Classification'].values
    
    i=0
    images=np.zeros([len(labels),dataset.iloc[0][1].shape[0],dataset.iloc[0][1].shape[1]])
    for image in dataset['Image']:
        images[i,...]=image
        i+=1      
        
    return [labels, images, dataset]