from keras.models import Sequential
from keras.layers import Dense
import config
from keras.callbacks import EarlyStopping
import pickle
from utils import *

class DenseNet:
    def __init__(self):
        self.model = self.init_model()

    def init_model(self):
        model = Sequential()
        for i, layer in enumerate(config.STRUCTURE_DENSE):
            if i == 0:
                model.add(Dense(layer[0], input_dim=28, activation=layer[1]))
            else:
                model.add(Dense(layer[0], activation=layer[1]))
        model.add(Dense(25, activation='relu'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def fit_model(self, x_train, y_train):
        x_train, self.minimums_x, self.maximums_x = normalize(x_train)
        y_train, self.minimums_y, self.maximums_y = normalize(y_train)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(x_train, y_train, epochs=3, batch_size=128,
                    validation_split=0.2, callbacks=[early_stopping])

    def save_model(self):
        with open('minimums_x.pickle', 'wb') as f:
            pickle.dump(self.minimums_x, f)
        with open('maximums_x.pickle', 'wb') as f:
            pickle.dump(self.maximums_x, f)
        with open('minimums_y.pickle', 'wb') as f:
            pickle.dump(self.minimums_y, f)
        with open('maximums_y.pickle', 'wb') as f:
            pickle.dump(self.maximums_y, f)
        self.model.save_weights("model_dense.h5")

    def load_model(self):
        with open('minimums_x.pickle', 'rb') as f:
            self.minimums_x = pickle.load(f)
        with open('maximums_x.pickle', 'rb') as f:
            self.maximums_x = pickle.load(f)
        with open('minimums_y.pickle', 'rb') as f:
            self.minimums_y = pickle.load(f)
        with open('maximums_y.pickle', 'rb') as f:
            self.maximums_y = pickle.load(f)
        self.model.load_weights("model_dense.h5")

    def get_data(self, data):
        data += self.minimums_x[-1]
        data /= self.maximums_x[-1]
        predict_data = self.model.predict(data)
        new_state, reward = (self.maximums_y[0] * predict_data[:, :24]) - self.minimums_y[0], (self.maximums_y[0] * predict_data[:, 24]) - self.minimums_y[0]
        return new_state, reward
