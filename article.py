import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# Data import and model definition
data = pd.read_excel('data_domain.xlsx')
model=RandomForestRegressor()
'''
model=SVR()
model=MLPRegressor()
'''
y=data['G']
X=data.drop('G',axis=1)
X,X_val,y,y_val=train_test_split(X,y,test_size=0.1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model.fit(X_train,y_train)

# Model validation
r_train,r_test,r_val=model.score(X_train,y_train),model.score(X_test,y_test),model.score(X_val,y_val)
pred_true=list(model.predict(X_train))
pred_test=list(model.predict(X_test))
pred_val=list(model.predict(X_val))
e_train,e_test,e_val=y_train-pred_true,y_test-pred_test,y_val-pred_val
m_etrain,std_etrain=np.mean(e_train), np.std(e_train)
m_etest,std_etest=np.mean(e_test), np.std(e_test)
m_eval,std_eval=np.mean(e_val), np.std(e_val)
rmse,mae=np.sqrt(mean_squared_error(y_val,pred_val)), mean_absolute_error(y_val,pred_val)

# Validation plot
figure = pyplot.figure(figsize = (10, 10))
pyplot.gcf().subplots_adjust(wspace=0.4, hspace = 0.4)
pyplot.figure(1)
pyplot.subplot(1, 2, 1)
pyplot.title('Training set')
pyplot.xlabel('True')
pyplot.ylabel('Predicted')
pyplot.text(150, 1700, 'r={}'.format(round(r_train,2)),color='red')
pyplot.scatter(y_train,pred_true)
pyplot.subplot(2, 2, 2)
pyplot.title('Testing set')
pyplot.xlabel('True')
pyplot.ylabel('Predicted')
pyplot.scatter(y_test,pred_test)
pyplot.text(180, 1500, 'r={}'.format(round(r_test,2)),color='red')
pyplot.subplot(2, 2, 4)
pyplot.title('Validation set')
pyplot.xlabel('True')
pyplot.ylabel('Predicted')
pyplot.scatter(y_val,pred_val)
pyplot.text(250, 1500, 'r={}'.format(round(r_val,2)),color='red')
figure = pyplot.figure(figsize = (10, 10))
pyplot.gcf().subplots_adjust(wspace=0.4, hspace = 0.4)
pyplot.subplot(1, 2, 1)
pyplot.title('Training set error')
pyplot.hist(e_train,color = 'yellow',edgecolor = 'red')
pyplot.subplot(2, 2, 2)
pyplot.title('Testing set error')
pyplot.hist(e_test,color = 'yellow',edgecolor = 'red')
pyplot.text(180, 1500, 'r={}'.format(round(r_test,2)),color='red')
pyplot.subplot(2, 2, 4)
pyplot.title('Validation set error')
pyplot.hist(e_val,color = 'yellow',edgecolor = 'red')
pyplot.text(250, 1500, 'r={}'.format(round(r_val,2)),color='red')

# Grid prediction
data2 = pd.read_excel('grid.xlsx')
data2.head()
d2=model.predict(data2)
data2['G (RF)']=d2
data2.to_excel('results.xlsx')
