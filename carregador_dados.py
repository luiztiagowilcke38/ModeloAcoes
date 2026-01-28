import yfinance as yf
import pandas as pd
import numpy as np

def obter_dados_b3(tickers, data_inicio="2020-01-01", data_fim=None):
    """
    Baixa dados históricos de ações da B3.
    
    Args:
        tickers (list): Lista de símbolos (ex: ['VALE3.SA', 'PETR4.SA']).
        data_inicio (str): Data de início no formato 'AAAA-MM-DD'.
        data_fim (str): Data final no formato 'AAAA-MM-DD'.
        
    Returns:
        pd.DataFrame: Preços de fechamento ajustados.
    """
    print(f"Baixando dados para: {tickers}")
    dados_brutos = yf.download(tickers, start=data_inicio, end=data_fim)
    print("Colunas retornadas:", dados_brutos.columns)
    
    if 'Adj Close' in dados_brutos.columns:
        dados = dados_brutos['Adj Close']
    elif 'Close' in dados_brutos.columns:
        dados = dados_brutos['Close']
    else:
        # Tenta acessar nível superior se multi-level
        try:
             dados = dados_brutos['Adj Close']
        except KeyError:
             print("Aviso: 'Adj Close' não encontrado. Tentando 'Close'.")
             dados = dados_brutos['Close']

    # Preencher valores ausentes
    dados = dados.ffill().bfill()
    
    return dados

def calcular_retornos_log(precos):
    """
    Calcula retornos logarítmicos a partir dos preços.
    
    Args:
        precos (pd.DataFrame): Dados de preço.
        
    Returns:
        pd.DataFrame: Retornos logarítmicos.
    """
    return np.log(precos / precos.shift(1)).dropna()

if __name__ == "__main__":
    # Teste simples
    lista_tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']
    dados_teste = obter_dados_b3(lista_tickers)
    print("Dados baixados (primeiras 5 linhas):")
    print(dados_teste.head())
    
    retornos_teste = calcular_retornos_log(dados_teste)
    print("\nRetornos Logarítmicos (primeiras 5 linhas):")
    print(retornos_teste.head())
