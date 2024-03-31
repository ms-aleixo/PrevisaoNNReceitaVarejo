from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

def build_model(input_shape, hyperparameters):
    model = Sequential([
        Dense(units=hyperparameters['units'], activation=hyperparameters['activation'], input_shape=(input_shape,)),
        Dense(units=1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=hyperparameters['learning_rate']), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model
