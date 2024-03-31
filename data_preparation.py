# src/data_preparation.py

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(database_path):
    conn = sqlite3.connect(database_path)
    query = """
    SELECT
        strftime('%Y-%m', month_year) AS month_year,
        local_id,
        atividade_pcm_id,
        value AS volume_vendas,
        (SELECT value FROM indice_receita_nominal WHERE local_id = v.local_id AND atividade_pcm_id = v.atividade_pcm_id AND month_year = v.month_year) AS receita_nominal
    FROM
        indice_volume_vendas AS v
    ORDER BY month_year, local_id, atividade_pcm_id;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_data(df):
    df['receita_nominal'] = pd.to_numeric(df['receita_nominal'], errors='coerce')
    df['volume_vendas'] = pd.to_numeric(df['volume_vendas'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    df[['volume_vendas', 'receita_nominal']] = imputer.fit_transform(df[['volume_vendas', 'receita_nominal']])
    X = df[['volume_vendas']]
    y = df['receita_nominal']
    return train_test_split(X, y, test_size=0.2, random_state=42)
