#Regression using Sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#
#Initiate x and y

x1=[]
y1=[]

for i in range(20):
    r=np.random.randint(20)
    x1.append(r)

print(x1)
y1=[]       #Initiate Y list
for i in range(20):
    r=np.random.randint(25)
    y1.append(r)

print(y1)

#Create DataFrame using the above lists
df=pd.DataFrame(zip(x1,y1),columns=["x","y"])
print(df)

#Convert list to arrays

x=np.array(x1).reshape(-1,1)
y=np.array(y1).reshape(-1,1)

#Prepare Train and Test Data
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
print(len(x_train))
print(len(x_test))

#LR
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
m=clf.coef_ #slope
c=clf.intercept_    #intercept
print("Slope = {}, intercept={}".format(m,c))


y_pred = clf.predict(x_test)    #predicting data
print("Accuracy = ",clf.score(x_train,y_train))

yhat=x*m[0]+c   #y = mx + c
print(yhat)
plt.scatter(x,y)    #Scatter plotting
plt.plot(x,yhat,lw=2,c='orange',label='Actuals vs Regression Line')   #straight line
#plt.title("Residual Plot")
plt.xlabel("Independent Variable(X)")
plt.ylabel("Dependent Variable(Y)")
plt.show()

#Plot residuals
"""
Residual Plots
Residuals are the difference between the dependent variable (y)
 and the predicted variable (y_predicted).
"""
residuals=y-yhat
plt.plot(x,residuals, 'o', color='darkblue')
plt.title("Residual Plot")
plt.xlabel("Independent Variable")
plt.ylabel("Residual")
plt.show()
