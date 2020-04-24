#Plot Linear Regression Line and find out Regressors count

#Create Lists for X and Y coordinates with random numbers

import numpy as np
import matplotlib.pyplot as plt

x1=[] #Initiate X List
for i in range(10):
    r=np.random.randint(10)
    x1.append(r)

print(x1)
y1=[]       #Initiate Y list
for i in range(10):
    r=np.random.randint(20)
    y1.append(r)
print(y1)

import pandas as pd
plt.scatter(x1,y1)  #Plot Scatter Plotting using x1 and y1 lists
#
#   Now draw Line using y=mx+c connecting these points
#   Also find out how points we can connect
#   Keep tracking Regressors
#
any_matched = False
for i in range(10):
    #yhat = mx + c here c=1.5, m=i+1
    print("\n\tIteration {} begins".format(i+1))
    yhat=list(map(lambda x: ((i+1)*x + 1.5),x1)) #Predicting values with Linear Regression line
    yhat = [int(j) for j in yhat]   #converting to integers
    print("\nActual=",y1)
    print("\nPredicted=",yhat)
    plt.plot(x1,yhat,lw=2)  #Plotting Straight line
    #Find the diff between actual and predicted
    y1s = pd.Series(y1) #Convert list to Series
    yhats = pd.Series(yhat)
    diff = y1s - yhats #Find diff actual vs predicted for Regressors counting
    print("\n\t Iteration {} results".format(i+1))
#
#   Print min and max details
#
#
#Compute actual vs predicted
#
    print("\nmin = ",diff.min())
    print("\nmax = ",diff.max())
    print("\nmean = ",diff.mean())
    print("\nstd =",diff.std())
#
#Compute Residuals
#
    print("\nCount those matched=",list(diff).count(0)) #how many it could touch
    print("\n\nResidual count = ",len(diff)-list(diff).count(0))
    plt.pause(1)

    if (list(diff).count(0)) > 0:
            print("\nIteration {} could hit somepoints {}: at index {}".format(i+1,list(diff).count(0),list(diff).index(0)))
            any_matched = True
            last_hit = i + 1
plt.show()

if(any_matched):
    print("\n Check last iteration",last_hit)
else:
    print("\n No iteration could hit atleast one")

#Keep changing m and c and find results
