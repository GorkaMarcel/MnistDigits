# MnistDigits
Kod do wykonania predykcji na zbiorze danych MNIST Digits 

Best neural net configuration in my case was:

conv(1,3,(3,3),(1,1)(1,1)) - conv(3,5,(3,3),(1,1)(1,1)) - 512 TanH - 256 Sigmoid - 128 LeakyReLu - 64 TanH)

For conv layers its 2 conf with hiperparamters given as above. Hidden layers parameters were given in num of - neurons in first layer tf of forst layer - ...... - num of neurons in last layer tf of last layer

I highly encourage you to check my work at Kaggle.com (link to MNIST Digits notebook: https://www.kaggle.com/code/cosmiccat/mnist-numbers-first-approach) where I have decribed what and why I did in my code.
