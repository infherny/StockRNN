# Recurrent Neural Networks


# Part 1 - Data preparation


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# for MacOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# training data

dataset_train = pd.read_csv('hexo_train.csv')
training_set = dataset_train[["Open", "Volume", "TSX_Open"]].values

# Feature scaling

sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(training_set)
# Create structure with 60 timeSteps and one output

y_train = []
for i in range(60, training_set.shape[0] - 5):
    y = []
    for j in range(0, 5):
        y.append(training_set_scaled[i+j, 0])
    y, np.array(y)
    y_train.append(y)
y_train = np.array(y_train)

# y_train = np.swapaxes(np.swapaxes(y_train, 0, 1), 1, 2)

# y_train = np.expand_dims(y_train, axis=0)


X_train = []
for j in range(0, 3):
    X = []
    for i in range(60, training_set.shape[0] - 5):
        X.append(training_set_scaled[i-60:i, j])
    X, np.array(X)
    X_train.append(X)
X_train = np.array(X_train)

# Reshaping

X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)

# Part 2 - Create RNN

regressor = Sequential()

# LSTM layer + Dropout

regressor.add(LSTM(units=50, return_sequences=True,
                   input_shape=(X_train.shape[1], 3)))
regressor.add(Dropout(0.2))

# LSTM 2

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# LSTM 3

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# LSTM 4

regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(0.2))

# output layer

regressor.add(Dense(units=5))

# Compilation

regressor.compile(optimizer="adam", loss="mean_squared_error")

# regressor.load_weights("regressor_weight.h5")

# Training

regressor.fit(X_train, y_train, epochs=500, batch_size=10)

# save

regressor.save_weights('regressor_weight.h5')

# Part 3 - Prediction

# testing data

dataset_test = pd.read_csv('hexo_test.csv')
real_stock_price = dataset_test[["Open", "Volume", "TSX_Open"]].values

# Prediction

X_test = []
for j in range(0, 3):
    X = []
    X.append(training_set_scaled[-60:, j])
    X, np.array(X)
    X_test.append(X)
X_test = np.array(X_test)
X_test = np.swapaxes(np.swapaxes(X_test, 0, 1), 1, 2)

X_test = X_test

predicted_stock_price = regressor.predict(X_test)

# il faut créer une matrice de transformation bidon

scFirstColunm = MinMaxScaler(feature_range=(0, 1))

training_set_open = dataset_train[["Open"]].values
training_set_scaled_first_colunm = \
    scFirstColunm.fit_transform(training_set_open)

predicted_stock_price = scFirstColunm.inverse_transform(predicted_stock_price)
predicted_stock_price = np.swapaxes(predicted_stock_price, 0, 1)

# graph

real_stock_price = dataset_test[["Open"]].values
real_stock_price = real_stock_price[0:5]

plt.plot(real_stock_price, color="red", label="real price")
plt.plot(predicted_stock_price, color="blue", label="predicted price")

plt.title("HEXO stock prediction")
plt.xlabel("day")
plt.ylabel("price")
plt.legend("")
plt.show()
