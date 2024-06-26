import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('housePrice.csv')

X=np.array(df['Size'])/1000
y=np.array(df.drop(columns=['Size']))/1000

X=np.c_[np.ones(X.shape[0]),X]

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)

theta=np.zeros((1,X.shape[1]))

convergenceDif=0.1
learningRate=0.2

def h(x,theta):
    return theta.dot(x.T)

def gradientDescent(X, y, learningRate):
    m = len(y)
    convergence = False

    while not convergence:
        theta_prev = theta.copy()
        for j in range(len(theta[0])):
            theta_sum = sum((h(X[i], theta) - y[i]) * X[i][j] for i in range(m))
            theta[0][j] -= (learningRate / m) * theta_sum

        if np.linalg.norm(theta - theta_prev) < convergenceDif:
            convergence = True

    return theta

theta=gradientDescent(X_train,y_train,learningRate)

y_pred=[h(i,theta) for i in X_test]

# for i in range(len(y_pred)):
#     print(f"Pentru dimensiunea de  a prezis {y_pred[i]} cand este {y_test[i]}")


X_train=np.delete(X_train,0,1)
plt.scatter(X_train,y_train)
plt.plot(X_test,y_pred)

plt.show()