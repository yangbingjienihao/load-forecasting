 ###采用2层小波分解，用前24个数据去预测第25个数据
 
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import tensorflow as tf
from tem import get_train_data
from keras.callbacks import TensorBoard

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix


path = '.\e2013.csv'
X_train0, y_train0, X_test0, y_test0, shifted_value = get_train_data(path)
X_train0 = np.reshape(X_train0, (X_train0.shape[0], X_train0.shape[1], 1))  
X_test0 = np.reshape(X_test0, (X_test0.shape[0], X_test0.shape[1], 1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(100, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(100, return_sequences=False))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation='linear'))
model.compile(loss="mse", optimizer="adam")
# evaluate the result
test_mse0 = model.evaluate(X_test0, y_test0, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse0, len(y_test0)))
# get the predicted values
predicted_values0 = model.predict(X_test0)
num_test_samples0 = len(predicted_values0)
predicted_values0 = np.reshape(predicted_values0, (num_test_samples0,1))
np.save("./gt.npy",(y_test0+shifted_value)*10000000)
np.save("./WLSTM_pred.npy", (predicted_values0+shifted_value)*10000000)
t = np.load("./gt.npy")
pred = np.load("./WLSTM_pred.npy")
# plot the results
fig = plt.figure()
plt.plot(t,c="g" )
plt.title('real load'))
plt.plot(pred,c="r" )
plt.title('prediction load')
plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e3)')
plt.show()


