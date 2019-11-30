Author: James Dickens
CSI 5155: Machine Learning, Course taught by Dr. Herna Viktor.

This is my code for the machine learning task of binary classification of the data available at 
http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD), which consists of 
weighted census data extracted from the 1994 and 1995 current population surveys conducted by the U.S. Census Bureau.
The goal is to evaluate five commonly used machine learning models (along with a semi-supervised neural network!) to 
classify whether a given instance makes more than 50K a year or not , a.k.a a binary classification task.

My code is organized as follows:

First Preprocess.py takes the initial census-income.data file and census-income.test file and 
- prints information about the data and its attributes
- removes duplicates from the training data
- deals with instance weight conflicts
- replaces missing values with their defaults
- writes the result to files:
'census-income.data/training_data_preprocess1', 
'census-income.test/testing_data_preprocess1'

Next Preprocess2.py 
- eliminates certain features
- simplifies certain features by using binning
- writes the result to the files:
'census-income.data/training_data_preprocess2', 
'census-income.test/testing_data_preprocess2'

Next Preprocess3.py 
- One-hot-encoding applied to categorical attributes:
  class of worker, education, enrolled education, married, race, sex, employment status, and tax filer status
- feature calibration of the occupation code and industry code
- Binarization of sex, country of birth of parents, country of birth of the person, and income category
- writes the result to the files:
'census-income.data/train_data',  
'census-income.test/test_data'

Models.py
- Trains and evalutes the five models on the training/testing set and prints evaluation metrics


