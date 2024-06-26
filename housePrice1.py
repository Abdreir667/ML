import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def h(x):
    return theta[1]*x

df=pd.read_csv("housePrice.csv")

X=np.array(df['Size'])
X_test=X[:int(len(X)*0.8)]
X_train=X[:int(len(X)*0.2):-1]
Y=np.array(df['Price'])
Y_test=Y[:int(len(Y)*0.8)]
Y_train=Y[:int(len(Y)*0.2):-1]

X_test = np.c_[np.ones(X_test.shape[0]), X_test]#adaugam o coloana plina cu 1 pe
#axa 0(axa liniilor ca est doar un vector) 
theta=np.linalg.inv(X_test.T.dot(X_test)).dot(X_test.T).dot(Y_test)#ecuatia normala
#pentru calcularea parametrilor
print(theta)
X_test=np.delete(X_test,0,1)#de pe axa 1 (axa coloanelor) stergem coloana 0

plt.scatter(X_test,Y_test)


Y_Pred=[h(i) for i in X_train]

plt.scatter(X_train,Y_train,color='red')
plt.scatter(X_train,Y_Pred,marker='x')

plt.show()

