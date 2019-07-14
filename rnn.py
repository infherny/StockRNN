# Recurrent Neural Networks

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pathlib import Path
import time

# for MacOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class StockRNN:

    def __init__(self, stock_name, recurrence=60):
        self.recurrence = recurrence
        self.model = Sequential()
        self.dataset_train = []
        self.dataset_test = []
        self.stock_name = stock_name

        self.y_train = []
        self.X_train = []

        self.training_set_scaled = []

    def create_model(self, lstm_layers=4):

        self.model.add(LSTM(units=64, return_sequences=True,
                            input_shape=(self.X_train.shape[1], 3),
                            dropout=0.3))

        for i in range(1, lstm_layers - 1):

            self.model.add(LSTM(units=64, return_sequences=True, dropout=0.3))

        self.model.add(LSTM(units=64, return_sequences=False, dropout=0.3))

        # output layer
        self.model.add(Dense(units=5))

        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def load_data(self, training_data, testing_data):

        self.dataset_train = pd.read_csv(training_data)
        self.dataset_test = pd.read_csv(testing_data)

        training_set = self.dataset_train[["Open", "Volume", "TSX_Open"]].values

        # Feature scaling

        sc = MinMaxScaler(feature_range=(0, 1))
        self.training_set_scaled = sc.fit_transform(training_set)

        self.y_train = []
        for i in range(self.recurrence, training_set.shape[0] - 5):
            y = []
            for j in range(0, 5):
                y.append(self.training_set_scaled[i + j, 0])
            y, np.array(y)
            self.y_train.append(y)
        self.y_train = np.array(self.y_train)

        self.X_train = []
        for j in range(0, 3):
            X = []
            for i in range(self.recurrence, training_set.shape[0] - 5):
                X.append(self.training_set_scaled[i - self.recurrence:i, j])
            X, np.array(X)
            self.X_train.append(X)
        self.X_train = np.array(self.X_train)

        # Reshaping
        self.X_train = np.swapaxes(np.swapaxes(self.X_train, 0, 1), 1, 2)

    def training(self, training_step, epochs=100, batch_size=5):

        weights = Path("regressor_weight.h5")
        if weights.is_file():
            self.model.load_weights('regressor_weight.h5')

        # boucle d'entrainement
        for i in range(0, training_step):
            self.model.fit(self.X_train, self.y_train, epochs=epochs,
                          batch_size=batch_size)
            self.model.save_weights('regressor_weight.h5')
            self.generate_plot()
            time.sleep(60 * 5)
            self.model.load_weights('regressor_weight.h5')
            time.sleep(10)

    def generate_plot(self):

        dataset_test = pd.read_csv('hexo_test.csv')
        real_stock_price = dataset_test[["Open", "Volume", "TSX_Open"]].values

        # Prediction

        X_test = []
        for j in range(0, 3):
            X = []
            X.append(self.training_set_scaled[-self.recurrence:, j])
            X, np.array(X)
            X_test.append(X)
        X_test = np.array(X_test)
        X_test = np.swapaxes(np.swapaxes(X_test, 0, 1), 1, 2)

        predicted_stock_price = self.model.predict(X_test)

        # il faut créer une matrice de transformation bidon
        sc_first_colunm = MinMaxScaler(feature_range=(0, 1))

        training_set_open = self.dataset_train[["Open"]].values
        training_set_scaled_first_colunm = \
            sc_first_colunm.fit_transform(training_set_open)

        predicted_stock_price = sc_first_colunm.inverse_transform(
            predicted_stock_price)
        predicted_stock_price = np.swapaxes(predicted_stock_price, 0, 1)

        # graph

        real_stock_price = dataset_test[["Open"]].values
        real_stock_price = real_stock_price[0:5]

        plt.plot(real_stock_price, color="red", label="real price")
        plt.plot(predicted_stock_price, color="blue", label="predicted price")

        plt.title(self.stock_name + " stock prediction")
        plt.xlabel("day")
        plt.ylabel("price")
        plt.legend(loc='upper left')
        plt.show()


# Exemple :

# hexo = StockRNN('Hexo', recurrence=120)
# hexo.load_data("hexo_train.csv", "hexo_test.csv")
# hexo.create_model(lstm_layers = 5)
# hexo.training(training_step=5)
