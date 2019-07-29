# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 00:28:12 2019

@author: LG
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


reg = LinearRegression()
log = LogisticRegression()


data=pd.read_csv(r'C:\Users\LG\Desktop\SeoulTech\Major\DataMining\Final\DataTest.csv')
data.head()

nameList = ['bogun_cd','gym','movie','art','region','age_20','age_30','age_40-65','sex','gungang','smoke','job','naBiman','exercise','biman','sleep','stress','life','marry']
name = data[nameList]

dataList=['bogun_cd','gym','movie','art','region','age','sex','gungang','smoke','job','naBiman','exercise','biman','sleep','stress','life','marry']


ageData_20 = data[['age_30','age_40-65','age_>=65']]
ageData_30 = data[['age_20','age_40-65','age_>=65']]
ageData_40 = data[['age_20','age_30','age_>=65']]
ageData_65 = data[['age_20','age_30','age_40-65']]

X = data[dataList]

age_X = ageData_20
#age_X = ageData_30
#age_X = ageData_40
#age_X = ageData_65


#'age_20','age_30','age_40-65','age_>=65'
X_vif = X
VIF_data = np.zeros((X_vif.shape[1], 2))

for i in range(X_vif.shape[1]) :
    VIF_X = X_vif.drop(X_vif.columns[[i]], axis = 1)
    VIF_x = X_vif.iloc[:,i]
    VIF_reg = reg.fit(VIF_X, VIF_x)
    vif = 1/ (1-VIF_reg.score(VIF_X, VIF_x))
    VIF_data[i,1] = vif
VIF_data = pd.DataFrame(VIF_data)
VIF_data.iloc[:,0] = dataList
VIF_data = VIF_data.rename(columns = {0: 'Variable',
                           1:'VIF'})
print("Variables : 'bogun_cd','gym','movie','art','region','age','sex','gungang','smoke','job','naBiman','exercise','biman','sleep','stress','life','marry'")
print(VIF_data)
print('--------------------------------------------------------------------------------------------')

y_vif = data[['drink_day','drink_cup']]
y_VIF_data = np.zeros((y_vif.shape[1], 2))

for i in range(y_vif.shape[1]) :
    VIF_Y = y_vif.drop(y_vif.columns[[i]], axis = 1)
    VIF_y = y_vif.iloc[:,i]
    VIF_reg = reg.fit(VIF_Y, VIF_y)
    vif = 1/ (1-VIF_reg.score(VIF_Y, VIF_y))
    y_VIF_data[i,1] = vif
y_VIF_data = pd.DataFrame(y_VIF_data)
y_VIF_data.iloc[:,0] = ['drink_day','drink_cup']
y_VIF_data = y_VIF_data.rename(columns = {0: 'Variable',
                           1:'VIF'})
print("Variables : 'drink_day','drink_cup'")
print(y_VIF_data)
print('--------------------------------------------------------------------------------------------')

y1=data['drink_day']
y2=data['drink_cup']


#y1 과 y2 바이너리 변환 고음주
y1_bin=1*np.logical_and(y1>=4,y1!=8)
y2_bin=1*np.logical_and(y2>=4,y2!=8)


y1_bin.hist(alpha = 0.7)
y2_bin.hist(alpha = 0.7)



x1_train, x1_test, y1_train, y1_test = train_test_split(X, y1_bin, 
                                                        test_size=0.3, random_state=0)
x2_train, x2_test, y2_train, y2_test = train_test_split(X, y2_bin, 
                                                        test_size=0.3, random_state=0)


age_X_train, age_X_test, y1_train, y1_test = train_test_split(age_X, y1_bin, 
                                                        test_size=0.3, random_state=0)
age_X_train, age_X_test, y2_train, y2_test = train_test_split(age_X, y2_bin, 
                                                        test_size=0.3, random_state=0)



#for i in ['bogun_cd','gym','movie','art','region','age','sex','gungang','smoke','exercise','brkfst','biman','sleep','stress','life','marry']:
#    #plt.scatter(x2_train[i], y2_train)
#    plt.scatter(data[i],y1_bin)
#    plt.xlabel(i)
##    plt.ylabel("drink_cup")
#   plt.show()



log.fit(x1_train, y1_train)
log.fit(x2_train, y2_train)
log.fit(age_X_train, y2_train)


log.score(x1_train, y1_train)
log.score(x1_test, y1_test)
log.score(age_X_test, y1_test)


log.score(x2_train, y2_train)
log.score(x2_test, y2_test)
log.score(age_X_test, y2_test)

import statsmodels.api as sm


logit1 = sm.Logit(y1_train, x1_train)
logit2 = sm.Logit(y2_train, x2_train)
logit3 = sm.Logit(y2_train, age_X_train)


result1 = logit1.fit()
result1.summary2()


result2 = logit2.fit()
result2.summary2()

result3 = logit3.fit()
result3.summary2()

np.exp(result1.params)
np.exp(result2.params)
np.exp(result3.params)




#sns.scatterplot(x='biman',y='drink_cup',data=name)
#plt.grid(True)
#plt.show()

