
#Step1数据预处理
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/4,random_state = 0)

#Step2训练集使用简单线性回归模型来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)#对训练集进训练

#Step3预测结果
Y_pred = regressor.predict(X_test)#使用训练得到的估计其对X_test进行预测

#Step4可视化

#训练集结果可视化
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.show()

'''
plt.scatter() 散点图画法，用法https://blog.csdn.net/m0_37393514/article/details/81298503
plt.plot(x,y,format_string,**kwargs) 用法https://blog.csdn.net/u014539580/article/details/78207537
'''

#测试集结果可视化
plt.scatter(X_test,Y_test,color = 'yellow')
plt.plt(X_test,regressor.predict(X_test),color = 'green')
plt.show()