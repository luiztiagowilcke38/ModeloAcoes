import numpy as np
from scipy.stats import levy_stable
import pandas as pd

class ModeloEstocasticoWilcke:
    """
    Implementa a Equação Diferencial Estocástica de Wilcke:
    dSt = theta * (mi - St) * dt + sigma * St * dL(alpha, beta)
    
    Combina Reversão à Média (Ornstein-Uhlenbeck) com Ruído Estável de Lévy 
    para capturar caudas pesadas (cisnes negros) e efeitos de memória.
    """
    
    def __init__(self, theta=0.1, mi=0.0, sigma=0.2, alpha=1.7, beta=0.0):
        """
        Args:
            theta (float): Velocidade de reversão à média.
            mi (float): Nível médio de longo prazo.
            sigma (float): Escala de volatilidade.
            alpha (float): Parâmetro de estabilidade (0 < alpha <= 2). 2 = Gaussiano, <2 = caudas pesadas.
            beta (float): Parâmetro de assimetria (-1 <= beta <= 1).
        """
        self.theta = theta
        self.mi = mi
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        
    def simular(self, S0, T, dt, simulacoes=1):
        """
        Simula o processo.
        
        Args:
            S0 (float): Preço inicial.
            T (float): Horizonte de tempo (anos).
            dt (float): Passo de tempo.
            simulacoes (int): Número de caminhos.
            
        Returns:
            np.ndarray: Caminhos simulados (passos x simulacoes).
        """
        N = int(T / dt)
        t = np.linspace(0, T, N)
        S = np.zeros((N, simulacoes))
        S[0] = S0
        
        for i in range(1, N):
            # Ruído de Lévy: dL ~ Stable(alpha, beta, 0, dt^(1/alpha))
            # A escala precisa ser dt^(1/alpha) para estabilidade da parametrização padrão
            escala = dt ** (1/self.alpha)
            dL = levy_stable.rvs(self.alpha, self.beta, loc=0, scale=escala, size=simulacoes)
            
            # Passo Anterior
            S_ant = S[i-1]
            
            # Deriva (Drift): Reversão à Média
            deriva = self.theta * (self.mi - S_ant) * dt
            
            # Difusão: Ruído de Cauda Pesada
            # Usando aproximação geométrica simples: S_t * (deriva + sigma * dL)
            difusao = self.sigma * S_ant * dL
            
            # Atualização
            S[i] = S_ant + deriva + difusao
            
            # Garantir preços não negativos (barreira absorvente em 0 ou pequeno epsilon)
            S[i] = np.maximum(S[i], 0.01)
            
        return t, S

class MovimentoBrownianoGeometrico:
    """MBG padrão para comparação."""
    def __init__(self, mi, sigma):
        self.mi = mi
        self.sigma = sigma
        
    def simular(self, S0, T, dt, simulacoes=1):
        N = int(T / dt)
        t = np.linspace(0, T, N)
        S = np.zeros((N, simulacoes))
        S[0] = S0
        
        for i in range(1, N):
            dW = np.random.normal(0, np.sqrt(dt), simulacoes)
            S[i] = S[i-1] * np.exp((self.mi - 0.5 * self.sigma**2) * dt + self.sigma * dW)
            
        return t, S
