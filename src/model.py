from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

def build_model(input_shape, hyperparameters):
    # Constrói o modelo de rede neural
    model = Sequential([
        Dense(units=hyperparameters['units'], activation=hyperparameters['activation'], input_shape=(input_shape,)),
        Dense(units=1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=hyperparameters['learning_rate']), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

def plot_model_performance_with_linear_regression(model_path, X_train, X_test, y_train, y_test):
    # Carrega o modelo
    model = load_model(model_path)
    predictions = model.predict(X_test).flatten()

    # Combina os dados de treinamento e teste para análise de regressão linear
    X_combined = np.vstack((X_train, X_test)).flatten()
    y_combined = np.concatenate((y_train, y_test))

    # Realiza regressão linear
    slope, intercept, r_value, p_value, std_err = linregress(X_combined, y_combined)
    y_pred_linreg = slope * X_combined + intercept
    residuals = y_combined - y_pred_linreg
    std_dev = np.std(residuals)

    # Plota os dados, as previsões e a regressão linear
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Dados de Treinamento')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='Dados de Teste')
    plt.scatter(X_test, predictions, color='red', alpha=0.5, label='Previsões')
    plt.plot(X_combined, y_pred_linreg, color='purple', label='Regressão Linear Simples', linewidth=2)

    # Plota bandas de desvio padrão da regressão linear
    plt.fill_between(X_combined, y_pred_linreg - std_dev, y_pred_linreg + std_dev, color='purple', alpha=0.2, label='Desvio Padrão')

    plt.xlabel("Índice de Volume de Vendas")
    plt.ylabel("Índice de Receita Nominal")
    plt.legend()
    plt.show()

    # Calcula e imprime as métricas de avaliação
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    print(f"Avaliação da Regressão Neural - MAE: {mae:.2f}, MSE: {mse:.2f}")
    print(f"Avaliação da Regressão Linear - MAE: {mean_absolute_error(y_combined, y_pred_linreg):.2f}, MSE: {mean_squared_error(y_combined, y_pred_linreg):.2f}, Desvio Padrão: {std_dev:.2f}")
