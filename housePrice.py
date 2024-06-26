import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

df=pd.read_csv('housePrice.csv')

X=df.drop(columns=['Price'])
Y=df['Price']

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.8)
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,y_pred,color='blue')

plt.xticks(()),plt.yticks(())

plt.show()

X_in=int(input("Dati dimensiunea unei case pentru a-i prezice pretul:"))
X_in=np.array(X_in).reshape(-1,1)
print(model.predict(X_in))

