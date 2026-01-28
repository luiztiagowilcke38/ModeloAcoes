import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from carregador_dados import obter_dados_b3, calcular_retornos_log
from modelos.modelos_matematicos import ModeloEstocasticoWilcke
from modelos.series_temporais import ajustar_tendencia_ar, estimar_volatilidade, ajustar_parametros_levy

def principal():
    # Configuração
    if not os.path.exists('saida'):
        os.makedirs('saida')
        
    lista_tickers = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']
    data_inicio = "2020-01-01"
    data_fim = datetime.now().strftime("%Y-%m-%d") # Usar data atual
    
    # 1. Carregar Dados
    print("--- Iniciando Coleta de Dados ---")
    precos = obter_dados_b3(lista_tickers, data_inicio=data_inicio, data_fim=data_fim)
    
    # 2. Processar e Prever por Ticker
    print("\n--- Iniciando Análise e Previsão (2026-2030) ---")
    
    # Horizonte de Previsão: De agora até 31-12-2030
    # Assumindo que 'agora' é inicio de 2026 com base no contexto do prompt
    data_atual = pd.to_datetime(data_fim)
    data_alvo = pd.to_datetime("2030-12-31")
    dias_para_prever = (data_alvo - data_atual).days
    anos_para_prever = dias_para_prever / 365.25
    
    dt = 1/252 # Passos diários
    
    plt.style.use('dark_background')
    
    for ticker in lista_tickers:
        print(f"Processando {ticker}...")
        
        # Obter série individual
        precos_ticker = precos[ticker].dropna()
        retornos = calcular_retornos_log(precos_ticker)
        
        # Estimar Parâmetros
        deriva = ajustar_tendencia_ar(precos_ticker)
        vol = estimar_volatilidade(retornos)
        alpha, beta, loc, scale = ajustar_parametros_levy(retornos)
        
        print(f"  Deriva (anual): {deriva:.4f}")
        print(f"  Volatilidade (anual): {vol:.4f}")
        print(f"  Levy Alpha: {alpha:.4f}, Beta: {beta:.4f}")
        
        # Inicializar Modelo
        # Modelo Wilcke simulando desvios com tendência e ruído pesado
        modelo = ModeloEstocasticoWilcke(
            theta=0.5, # Reversão à média moderada
            mi=deriva, # Direção da tendência
            sigma=vol,
            alpha=alpha,
            beta=beta
        )
        
        # Simulação
        S0 = precos_ticker.iloc[-1]
        t_sim, S_sim = modelo.simular(S0, anos_para_prever, dt, simulacoes=50)
        
        # Criar Datas para Simulação
        datas_sim = [data_atual + timedelta(days=d*365.25) for d in t_sim]
        
        # Plotagem
        plt.figure(figsize=(12, 6))
        
        # Plotar Histórico (Últimos 2 anos apenas para clareza)
        inicio_historico = data_atual - timedelta(days=730)
        dados_hist = precos_ticker[precos_ticker.index > inicio_historico]
        plt.plot(dados_hist.index, dados_hist.values, label='Histórico', color='cyan', linewidth=1.5)
        
        # Plotar Simulações
        # Caminho médio
        caminho_medio = np.mean(S_sim, axis=1)
        plt.plot(datas_sim, caminho_medio, label='Previsão Média (Modelo Wilcke)', color='yellow', linewidth=2)
        
        # Plotar alguns caminhos individuais
        for i in range(min(10, S_sim.shape[1])):
            plt.plot(datas_sim, S_sim[:, i], color='white', alpha=0.1)
            
        # Intervalos de Confiança (Verificação de caudas pesadas)
        limite_inferior = np.percentile(S_sim, 5, axis=1)
        limite_superior = np.percentile(S_sim, 95, axis=1)
        plt.fill_between(datas_sim, limite_inferior, limite_superior, color='white', alpha=0.1, label='Intervalo de Confiança 90%')
        
        plt.title(f"Previsão de Preço: {ticker} (2026-2030) - Modelo Estocástico de Wilcke", fontsize=14)
        plt.xlabel("Data")
        plt.ylabel("Preço (R$)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        
        # Salvar Gráfico
        nome_arquivo = f"saida/previsao_{ticker.replace('.', '_')}.png"
        plt.savefig(nome_arquivo)
        plt.close()
        print(f"  Gráfico salvo em: {nome_arquivo}")

if __name__ == "__main__":
    principal()
