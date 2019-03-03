Steps to PetFinder!

Cleaning:
Data cleaning for train/test csv file

Modeling:
1.  Baseline: XGB/logistic for train file, 70/30 
2. CNN for the image, use the PetID to decide which is the train and test. Only use the train for CNN
        1.  
3.  RNN for the description, use the PetID to decide which is the train and test. Only use the train for RNN
4. Take the sentiment data and convert them to features. 
5. Build the XGB with 1, 2, 3, 4

Prediction: 
1. Predict on the test data. 

# Code to split train data to train and validation

```
import numpy as np
import pandas as pd
import glob
import re

np.random.seed(6)
train = pd.read_csv('../input/train/train.csv')
train = train.assign(train_mask = np.random.rand(len(train)) < 0.8)

train_images = [re.sub(".*/", "", x) for x in glob.glob("../input/train_images/*.jpg")]
train_images_df = pd.DataFrame({'file_name':train_images,
                                'PetID':[re.sub("-.*", "", x) for x in train_images]})
train_images_df = pd.merge(train_images_df, train[['PetID', 'AdoptionSpeed', 'train_mask']],
                           how='left', on='PetID')
train_images_df
```
