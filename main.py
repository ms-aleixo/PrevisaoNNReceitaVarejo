from src.data_preparation import load_data, prepare_data
from src.model import build_model, plot_model_performance_with_linear_regression
import src.optimization as optimization
import optuna
import os

def run_and_evaluate_full_df(database_path, model_save_directory):
    # Carrega e prepara os dados
    df = load_data(database_path)
    if df.empty:
        print("Nenhum dado disponível. Encerrando...")
        return
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Otimiza os hiperparâmetros do modelo usando Optuna
    print("Otimizando hiperparâmetros do modelo...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: optimization.objective(trial, X_train, y_train, X_test, y_test), n_trials=10)
    best_hyperparameters = study.best_trial.params
    print(f"Melhores Hiperparâmetros: {best_hyperparameters}")

    # Treina o modelo final com os melhores hiperparâmetros
    print("Treinando modelo final...")
    final_model = build_model(X_train.shape[1], best_hyperparameters)
    final_model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

    # Avalia o modelo final
    predictions = final_model.predict(X_test).flatten()
    mse = optimization.mean_squared_error(y_test, predictions)
    mae = optimization.mean_absolute_error(y_test, predictions)
    print(f"Avaliação do Modelo Final - MAE: {mae}, MSE: {mse}")

    # Salva o modelo final
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    model_save_path = os.path.join(model_save_directory, 'model_full_dataset.h5')
    final_model.save(model_save_path)
    print(f"Modelo salvo em {model_save_path}")

    # Plota a performance do modelo com regressão linear
    plot_model_performance_with_linear_regression(model_save_path, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    database_path = 'data/IBGE_tabela_8882.db'  # Ajuste o caminho conforme necessário
    model_save_directory = 'models/'  # Diretório para salvar o modelo treinado
    run_and_evaluate_full_df(database_path, model_save_directory)
