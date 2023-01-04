#library import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#training data set import
train = pd.read_csv('train.csv')
#training data set basc information
print(train.head())
#test data set import
test = pd.read_csv('test.csv')

#transforming data from pandas data frame to numpy and splitting them to y and x
#labels
y = train.label.to_numpy()

#training independet variables
xTrain = train.drop('label',axis=1).to_numpy()
#test independent variable
xTest = test.to_numpy()

#data cohesion checking
#y lenght check
print("\nNum of labels :" + str(len(y)))
#x col numbers
print("\nNumber of individuals in the training set :" + str(len(xTrain[:,1])))
#x rows numbers
print("\nThe number of features of an individual in the training set :" + str(len(xTrain[1,:])))
#test columns number
print("\nIość osobników w zbiorze testowyml :" + str(len(xTest[:,1])))
#test rows number
print("\nThe number of characteristics of the individual in the test set :" + str(len(xTest[1,:]))+'\n')
#bottom of the test data frame to check number of individuals
print(test.tail(5))

#ploting images no. 0,1,3
plt.imshow(xTrain[2,:].reshape(28,28),cmap="gray")
plt.show()
plt.imshow(xTrain[1,:].reshape(28,28),cmap="gray")
plt.show()
plt.imshow(xTrain[3,:].reshape(28,28),cmap="gray")
plt.show()

#saving numpy arrays to external files
np.save("xTrain.npy", xTrain)
np.save("xTest.npy", xTest)
np.save("labels.npy", y)



