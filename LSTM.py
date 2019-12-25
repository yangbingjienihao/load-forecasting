from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

# define a function to convert a vector of time series into a 2D matrix
def convertSeriesToMatrix(vectorSeries, sequence_length):
    matrix=[]
    for i in range(len(vectorSeries)-sequence_length+1):
        matrix.append(vectorSeries[i:i+sequence_length])
    return matrix


np.random.seed(1234)


df_raw = pd.read_csv('.\e2013.csv', header=None)

# numpy array
df_raw_array = df_raw.values
list_hourly_load = [df_raw_array[i,:] /10000000 for i in range(0, len(df_raw)) ]
list_hourly_load =np.array(list_hourly_load)
list_hourly_load = list_hourly_load.reshape(-1)
# the length of the sequnce for predicting the future value
sequence_length = 25
# convert the vector to a 2D matrix
matrix_load = convertSeriesToMatrix(list_hourly_load, sequence_length)
matrix_load = np.array(matrix_load)
shifted_value = matrix_load.mean()  
matrix_load -= shifted_value   
print ("Data  shape: ", matrix_load.shape)
train_row = int(round(0.98 * matrix_load.shape[0]))
train_set = matrix_load[:train_row, :]
X_train = train_set[:, :-1]
y_train = train_set[:, -1] 
X_test = matrix_load[train_row:, :-1]
y_test = matrix_load[train_row:, -1]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# build the model
model = Sequential()
model.add(LSTM( input_dim=1, output_dim=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(output_dim=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(output_dim=1, activation='linear'))
model.compile(loss="mse", optimizer="rmsprop")
# train the model
model.fit(X_train, y_train, batch_size=512, nb_epoch=200, validation_split=0.05, verbose=1)
test_mse = model.evaluate(X_test, y_test, verbose=1)
print ('\nThe mean squared error (MSE) on the test data set is %.3f over %d test samples.' % (test_mse, len(y_test)))
predicted_values = model.predict(X_test)
num_test_samples = len(predicted_values)
predicted_values = np.reshape(predicted_values, (num_test_samples,1))
np.save("./gt.npy", (y_test+shifted_value)*10000000)
np.save("./LSTM_pred.npy", (predicted_values+shifted_value)*10000000)
t = np.load("./gt.npy")
pred = np.load("./LSTM_pred.npy")

fig = plt.figure()
plt.plot(t, c="g" )
plt.plot(pred, c="r" )


plt.xlabel('Hour')
plt.ylabel('Electricity load (*1e5)')
plt.show()
fig.savefig('output_load_forecasting.jpg', bbox_inches='tight')

