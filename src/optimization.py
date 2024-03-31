import optuna
from .model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

def objective(trial, X_train, y_train, X_val, y_val):
    # Define o objetivo para a otimização com Optuna
    hyperparameters = {
        'units': trial.suggest_categorical('units', [32, 64, 128, 256]),
        'activation': trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh']),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    }
    model = build_model(X_train.shape[1], hyperparameters)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stopping, reduce_lr], verbose=0)
    loss, mae = model.evaluate(X_val, y_val, verbose=0)

    # Faz previsões no conjunto de validação para calcular o MSE
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)

    print(f"Métricas de Avaliação - MAE: {mae}, Loss: {loss}, MSE: {mse}")
    return mae  # Optuna otimiza com base neste valor
