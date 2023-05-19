import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os

GRAPH_FOLDER = './graph'

def getFullFilePath(filename):
    return os.path.join(GRAPH_FOLDER, filename)

def splitData(data, train_size):
    X = data.iloc[:,:-1]
    Y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, train_size=train_size/100)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # one_hot = OneHotEncoder()
    # y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    # y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

    return X_train_scaled, X_test_scaled, y_train, y_test