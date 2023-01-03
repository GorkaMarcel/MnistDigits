import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import time
import torch.nn.functional as F

#setting device
if torch.cuda.is_available():
    print('Cuda')
    dev = "cuda"
else:
    print('CPU')
    dev = "cpu"
device = torch.device(dev)

#import data
xTrain = np.load('xTrain.npy')
y = np.load('labels.npy')
xTestFinal = np.load('xTest.npy')

#train test split
xTrain,xTest,yTrain,yTest = train_test_split(xTrain,y,test_size=0.2)

#converting to tensor
xTestFinal = torch.from_numpy(xTestFinal.reshape(xTestFinal.shape[0],1,28,28))
xTrain = torch.from_numpy(xTrain.reshape(xTrain.shape[0],1,28,28))
yTrain = torch.from_numpy(yTrain)
xTest = torch.from_numpy(xTest.reshape(xTest.shape[0],1,28,28))
yTest = torch.from_numpy(yTest)

#changeing type due to errors
xTestFinal = xTestFinal.type(torch.float32)
xTrain = xTrain.type(torch.float32)
xTest = xTest.type(torch.float32)

#sending tensors to device
xTestFinal = xTestFinal.to(device)
xTrain = xTrain.to(device)
yTrain = yTrain.to(device)
xTest = xTest.to(device)
yTest = yTest.to(device)

#creating data set
trainDataSet = TensorDataset(xTrain,yTrain)
#definig batch size
batchSize = 512
#creating data loader
dataLoad = DataLoader(trainDataSet, batchSize, shuffle='True')

#neural net parameters
#offset
z=0
#number of layers
numOfLayers = 3
#defining input, hidden layers and output size, number of epochs
inputSize = 196
hiddenLayersSizes = [512,256,128,64,36,16,16,8,8,2,2,2,2,2]
outputSize = 10
numOfEpochs = 20
#learning rate
learningRate = 0.001
#loss function
lossFunction = nn.CrossEntropyLoss


#defining class model
class Model(nn.Module):
    def __init__(self,inputSize, hiddenLayersSizes, outputSize):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0))
        self.layer1 = nn.Linear(inputSize,hiddenLayersSizes[0+z])
        self.layer2 = nn.Linear(hiddenLayersSizes[0+z], hiddenLayersSizes[1+z])
        self.layer3 = nn.Linear(hiddenLayersSizes[1+z], hiddenLayersSizes[2+z])
        self.layer4 = nn.Linear(hiddenLayersSizes[2+z], hiddenLayersSizes[3+z])
        self.layer5 = nn.Linear(hiddenLayersSizes[(numOfLayers-1)+z], outputSize)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.flatten(x,1)
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.LeakyReLU()(x)
        x = self.layer3(x)
        x = nn.Tanh()(x)
        #x = self.layer4(x)
        #x = nn.Sigmoid()(x)
        x = self.layer5(x)
        x = nn.LogSoftmax(dim=1)(x)
        return x


#assigning model to variable
model = Model(inputSize,hiddenLayersSizes,outputSize)
model = model.to(device)
#optimizer
optimizer = torch.optim.Adamax(model.parameters(), lr = learningRate)

#learning process
lossHistory = [0]*(numOfEpochs+1)
accTrainHistory = [0]*(numOfEpochs+1)
accTestHistory = [0]*(numOfEpochs+1)
timeHist = [0]*(numOfEpochs+1)

for epoch in range(numOfEpochs):
    # time measure start
    start = time.time()
    for xBatch, yBatch in dataLoad:
        pred = model(xBatch)
        loss = lossFunction()(pred, yBatch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lossHistory[epoch] += loss.item()*yBatch.size(0)
    # time measure stop
    stop = time.time()
    timeHist[epoch] += (stop-start) * (10**3)
    predTrain = model(xTrain)
    accTrain = (torch.argmax(predTrain, dim=1)==yTrain).float().mean()
    lossHistory[epoch] /= len(dataLoad.dataset)
    predTest = model(xTest)
    accTest = (torch.argmax(predTest, dim=1)==yTest).float().mean()
    accTrainHistory[epoch+1] = accTrain.cpu()
    accTestHistory[epoch+1] = accTest.cpu()
    print("Epoka uczenia: " + str(epoch))
    print("Dokładność na zbiorze treningowym: " + str(accTrain))
    print("Dokładność na zbiorze testowym: " + str(accTest))
    print("Czas:    " + str((stop-start) * (10**3))+" ms\n")

#wyświtlanie rezultatu
plt.plot(accTrainHistory,lw=3)
plt.plot(accTestHistory,lw=3)
plt.title('Dokładność')
plt.xlabel('Epoka uczenia')
plt.ylabel('Dokładność')
plt.legend(['Zbiór Treningowy','Zbiór Testowy'])
plt.show()

#accuracy na zbiorze testowym
pred = model(xTest)
corr = (torch.argmax(pred,dim=1)==yTest).float()
acc = corr.mean()
print("Accuracy on test data set is equal to :  " + str(acc))
print("Training time is equal to :   " + str(sum(timeHist)) + " ms\n")

#Kaggle predictions
predFinal = model(xTestFinal)
corrFinal = (torch.argmax(predFinal,dim=1)).cpu().numpy()
submission = pd.read_csv(r'C:\Users\kosmi\PycharmProjects\MNISTInz\data\sample_submission.csv')
submission['Label'] = corrFinal.reshape(-1,1)
submission.to_csv('submission.csv',index=None)