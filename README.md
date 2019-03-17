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
# Data cleaning function for both train and test

```
train = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')
test = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')
breed_label = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
color_label = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
state_label = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

def clean(data, train_valid=True):
    if train_valid:
        data = data.assign(train_mask = np.random.rand(len(data)) < 0.8)
    
    data.Type = data.Type-1
    data.Gender = data.Gender-1
    data.Vaccinated = data.Vaccinated-1
    data.Dewormed = data.Dewormed-1
    data.Sterilized = data.Sterilized-1
    
    data = data.assign(NameLength = data.Name.str.len(),
                       description_length = data.Description.str.len())\
               .merge(breed_label[['BreedID', 'BreedName']], how='left', left_on='Breed1', right_on='BreedID')\
               .merge(color_label, how='left', left_on='Color1', right_on='ColorID')\
               .drop(columns=['Breed1', 'Breed2', 'BreedID', 'ColorID', 'Color1', 'RescuerID', 'Description'])\
               .rename(columns={'ColorName':'Color1'})\
               .merge(color_label, how='left', left_on='Color2', right_on='ColorID')\
               .drop(columns=['ColorID', 'Color2'])\
               .rename(columns={'ColorName':'Color2'})\
               .merge(color_label, how='left', left_on='Color3', right_on='ColorID')\
               .drop(columns=['ColorID', 'Color3'])\
               .rename(columns={'ColorName':'Color3'})\
               .merge(state_label, how='left', left_on='State', right_on='StateID')\
               .drop(columns=['StateID', 'State'])
    
    data.loc[data.Name.isin(['No Name', 'No Name Yet', 'Unknown']) | data.Name.isnull(), 'NameLength'] = 0
    data.loc[~data.BreedName.isin(['Mixed Breed', 'Domestic Short Hair',
                                   'Domestic Medium Hair', 'Tabby', 'Domestic Long Hair',
                                   'Siamese', 'Persian', 'Labrador Retriever','Shih Tzu',
                                   'Poodle', 'Terrier', 'Golden Retriever', 'Calico']) , 'BreedName'] = 'Other'
    data.loc[data.StateName.isin(['Kelantan', 'Labuan', 'Pahang', 'Sabah', 'Sarawak', 
                                  'Terengganu']), 'StateName'] = 'Other'
    data.loc[data.Gender == 2, 'Gender'] = np.NaN
    data.loc[data.Vaccinated == 2, 'Vaccinated'] = np.NaN
    data.loc[data.Dewormed == 2, 'Dewormed'] = np.NaN
    data.loc[data.Sterilized == 2, 'Sterilized'] = np.NaN
    
    data = data.drop(columns=['Name'])
    
    data_one_hot_encoding = pd.get_dummies(data, 
                                           prefix=['BreedName', 'Color1', 'Color2', 'Color3', 'StateName'], 
                                           columns=['BreedName', 'Color1', 'Color2', 'Color3', 'StateName'])\
                              .drop(columns='PetID')
    
    if train_valid:
        data_one_hot_encoding = data_one_hot_encoding.drop(columns='train_mask')
        one_hot_train = data_one_hot_encoding.loc[data.train_mask,]
        one_hot_valid = data_one_hot_encoding.loc[~data.train_mask,]
        d_matrix_train = xgb.DMatrix(data=one_hot_train.drop(columns='AdoptionSpeed'), 
                                     label=one_hot_train[['AdoptionSpeed']])
        d_matrix_valid = xgb.DMatrix(data=one_hot_valid.drop(columns='AdoptionSpeed'), 
                                     label=one_hot_valid[['AdoptionSpeed']])
        result = [data, d_matrix_train, d_matrix_valid]
    else:
        d_matrix = xgb.DMatrix(data_one_hot_encoding)
        result = [data, d_matrix]
        
    return(result)
    
train = clean(train)
test = clean(test, False)
```
