import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\Hanshu\Desktop\excel data_ML\Salary_Data.csv')

print('Data shape:', dataset.shape)

x = dataset.iloc[: , :-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m = regressor.coef_

c = regressor.intercept_

(m*12)+c

(m*20)+c

bias = regressor.score(x_train , y_train)
bias

variance = regressor.score(x_test , y_test)
variance


# STATS
#MEAN
dataset.mean()       # this will give mean of entire dataframe 
dataset['Salary'].mean()     # this will give us mean of that particular column 
 
#MEDIAN
dataset.median()            # this will give median of entire dataframe 
dataset['Salary'].median()      # this will give us median of that particular column  

#MODE
dataset.mode()             
dataset['Salary'].mode()     # this will give us mode of that particular column 

#VARIANCE              
dataset.var()   # this will give variance of entire dataframe  
dataset['Salary'].var()       # this will give us variance of that particular column

# STANDARD DEVIATION
dataset.std()        # this will give standard deviation of entire dataframe  
dataset['Salary'].std()       # this will give us standard deviation of that particular column


#COEFFICIENT OF VARIATION(cv)
# for calculating cv we have to import a library first
from scipy.stats import variation

variation(dataset.values)  # this will give cv of entire dataframe
variation(dataset['Salary'])        # this will give us cv of that particular column


# CORRELATION
dataset.corr()   # this will give correlation of entire dataframe
dataset['Salary'].corr(dataset['YearsExperience'])   # this will give us correlation between these



#SKEWNESS
dataset.skew()     # this will give skewness of entire dataframe  
dataset['Salary'].skew()           # this will give us skewness of that particular colum



#STANDARAD ERROR
dataset.sem()     # this will give standard error of entire dataframe 
dataset['Salary'].sem()           # this will give us standard error of that particular colum


#Z-SCORE
 # for calculating Z-score we have to import a library firs
import scipy.stats as stats
dataset.apply(stats.zscore)      # this will give Z-score of entire dataframe   


stats.zscore(dataset['Salary'])   # this will give us Z-score of that particular column



# DEGREE OF FREEDOM
a = dataset.shape[0]               # this will gives us no.of rows
b = dataset.shape[1]             ] # this will give us no.of column

degree_of_freedom = a-b
print(degree_of_freedom)            # this will give us degree of freedom for entire datase


# SSR
y_mean = np.mean(y)               # this will calculate mean of dependent variabl
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)


#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)


#SST
mean_total = np.mean(dataset.values)    # here df.to_numpy()will convert pandas Dataframe to Num  
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


#R2       # R-Square
r_square = 1 - SSR/SST
print(r_square)









