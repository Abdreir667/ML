import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.setrecursionlimit(10001)

df=pd.read_csv("students.csv")
df['gre']=df['gre'] / 1000
X=np.array(df.drop(columns='admit'))
y=np.array(df['admit'])


X=np.c_[np.ones(X.shape[0]),X]#x[0]=1

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)


theta=np.array([[0 for i in range(X.shape[1])]])#inizialiam parametrii cu 0

recursionLimit=200
learningRate=0.1
tau=100

def w(x, xi, tau):
    return np.exp(-np.sum((x - xi) ** 2) / (2 * tau ** 2))

def h(x, theta):#ipoteza 
    z = theta.dot(x.T)
    return float(1 / (1 + np.exp(-z[0])))

def gradientAscent(X, y, learningRate, theta):  # calculam fiecare parametru
    m = len(y)
    for j in range(len(theta[0])):
        iterator = 0
        while True:
            theta_sum = 0
            for i in range(m):
                weight = w(X[i], X,tau)  #daca vrem sa fie regresie ponderata,atunci scoatem 
                #functia w-de ponderare din comentariu
                theta_sum += weight * (y[i] - h(X[i], theta)) * X[i][j]
            theta[0][j] += learningRate * theta_sum
            iterator += 1
            if recursionLimit<iterator:
                break
    return theta

theta = gradientAscent(X_train, y_train, learningRate, theta)

y_pred = [h(x, theta) for x in X_test]

corecte=0

for i in range(len(y_pred)):
    if y_pred[i]>=0.5 and y_test[i]==1.0:
        corecte+=1
    elif y_pred[i]<0.5 and y_test[i]==0.0:
        corecte+=1
        
print(corecte*100/float(len(y_pred)))



