# import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load the dataset
dataset = pd.read_csv(r'C:\Users\Hanshu\Downloads\Data.csv')

# we devided the dataset into X & Y
# Independent Variables(X)
X = dataset.iloc[:, :-1].values

# Dpendent variable(Y)
Y = dataset.iloc[:, 3].values

# impute --> is transformer fir missing values handling
from sklearn.impute import SimpleImputer

# Hndling missing numerical values,here used mean/median strategy, mode didn't work
imputer = SimpleImputer(strategy ='mean')


imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])
X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(Y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y, train_size=0.8, test_size=0.2)

# FEAUTURE SCALING

from sklearn.preprocessing import Normalizer
sc_X = Normalizer()

x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)


 