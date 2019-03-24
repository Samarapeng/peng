#day 1 数据预处理

#Step1 导入库
import numpy as np
import pandas as pd

#Step2 导入数据集
dataset = pd.read_csv('C:/Users/Shinelon/Desktop/Machine Learning 100days/day1/Data.csv')
X = dataset.iloc[ : , :-1].values #。iloc【行，列】
Y = dataset.iloc[ : , 3].values #全部行or列，【a】第a行or列 【a，b，c】第a，b，c行or列
'''.iloc将数据提取出来，而.values将数据整理成矩阵或者向量的形式'''

print(X)
print(Y)

#Step3处理丢失数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy= "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
'''
Imputer填补缺失值：sklearn.preprocessing.Imputer(missing_values=’NaN’, strategy=’mean’, axis=0, verbose=0, copy=True)
missing_values为缺失值，可以是整数或者NaN，默认为后者，也就是将数据集中的此项替换
strategy为填补策略，mean为均值替换，median为中位数替换，frequent为众数替换
axis为轴数 0为列 1为行
copy True为创建附件，在新建数据集上修改，False为直接在原有数据集中修改
'''

print(X)
print(Y)

#Step4解析分类数据
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : ,0] = labelencoder_X.fit_transform(X[ : ,0])

#LabelEncoder对不连续的数字或者文本（分类特征值）进行编号
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit([1,5,67,100])
le.transform([1,1,100,67,5])
输出： array([0,0,3,2,1])
'''

#创建虚拟变量
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
#OneHotEncode用于将分类的数据扩维
'''
ohe = OneHotEncoder()
ohe.fit([[1],[2],[3],[4]])
ohe.transform([2],[3],[1],[4]).toarray()
输出：[ [0,1,0,0] , [0,0,1,0] , [1,0,0,0] ,[0,0,0,1] ]
'''

#Step5拆分数据集为训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)
'''
X_train,X_test, y_train, y_test =train_test_split(train_data,train_target,test_size=0.3, random_state=0)
train_data:被划分的样本特征集
train_target：被划分的样本标签
test_size:0~1之间代表样本占比，整数代表样本数量
random_state：随机数种子
	如果是0，则每次得到的随机数组是不一样的，如果每次都是int A，那么其他条件不变的情况下，每次得到的都是同一组随机数组
'''
#Step6特征量化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
StandardScaler:转换为均值为0，方差为1的正态分布
'''