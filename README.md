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
