import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levy_stable

def ajustar_tendencia_ar(dados):
    """
    Ajusta uma tendência linear simples para estimar a deriva de longo prazo.
    """
    # Regressão linear simples nos preços logarítmicos
    y = np.log(dados.values)
    x = np.arange(len(y))
    inclinacao, intercepto, r_valor, p_valor, erro_padrao = stats.linregress(x, y)
    
    # Retornar deriva anualizada
    # Assumindo dados diários (252 dias)
    deriva_anualizada = inclinacao * 252 
    return deriva_anualizada

def estimar_volatilidade(retornos):
    """
    Estima a volatilidade anualizada.
    """
    return retornos.std() * np.sqrt(252)

def ajustar_parametros_levy(retornos):
    """
    Ajusta os parâmetros da Distribuição Estável de Lévy aos retornos.
    
    Returns:
        tuple: (alpha, beta, loc, scale)
    """
    # Ajuste heurístico rápido para evitar lentidão do levy_stable.fit
    # Alpha typico para mercado financeiro: 1.5 - 1.8
    # Beta typico: perto de 0
    # Usamos os dados para estimar Loc e Scale apenas, assumindo Alpha/Beta baseados na curtose
    
    # Estimativa de Alpha via Curtose (grosso modo)
    curtose = stats.kurtosis(retornos)
    # Curtose alta -> Alpha menor. Normal (k=0) -> Alpha=2. 
    # Mapeamento heurístico simples:
    alpha_est = max(1.1, min(1.95, 2.0 - 0.1 * np.log1p(curtose))) if curtose > 0 else 1.95
    
    # Beta zero por simplicidade (simetria)
    beta_est = 0.0
    
    # Loc e Scale via interquartil ou desvio
    loc_est = np.mean(retornos)
    scale_est = np.std(retornos) / np.sqrt(2) # Ajuste grosseiro para distribuição estável
    
    # Retorna parâmetros estimados
    return alpha_est, beta_est, loc_est, scale_est
