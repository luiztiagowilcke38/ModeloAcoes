import os

def gerar_leiaue():
    conteudo = r"""# Modelo Matemático-Estatístico de Previsão de Ações B3

**Autor:** Luiz Tiago Wilcke  
**Data:** Janeiro, 2026

## Visão Geral

Este projeto apresenta um novo modelo híbrido para a previsão de preços de ativos financeiros na Bolsa de Valores Brasileira (B3). O modelo combina análise de séries temporais clássica com componentes estocásticos avançados, focando especificamente na modelagem de **caudas pesadas** (heavy tails) observadas em mercados reais.

## Fundamentação Teórica

A abordagem tradicional (Movimento Browniano Geométrico) falha em capturar eventos extremos ("Cisnes Negros"). O modelo proposto por **Wilcke** endereça isso substituindo o movimento browniano gaussiano por um processo de Lévy Alpha-Estável.

### A Equação Diferencial Estocástica de Wilcke

A dinâmica do preço do ativo $S_t$ é modelada pela seguinte Equação Diferencial Estocástica (EDE):

$$
dS_t = \theta (\mu(t) - S_t) dt + \sigma(t) S_t dL_{\alpha, \beta}^{\gamma}
$$

Onde:

- **$S_t$**: Preço do ativo no tempo $t$.
- **$\theta$**: Parâmetro de velocidade de reversão à média.
- **$\mu(t)$**: Nível de equilíbrio ou tendência de longo prazo, ajustado dinamicamente via modelos lineares (ARIMA).
- **$\sigma(t)$**: Volatilidade estocástica dependente do tempo.
- **$dL_{\alpha, \beta}^{\gamma}$**: Incremento do Processo de Lévy Estável, caracterizado por:
    - $\alpha \in (0, 2]$: Índice de estabilidade. Valores $< 2$ geram distribuições de cauda pesada (leptocúrticas).
    - $\beta \in [-1, 1]$: Parâmetro de assimetria.

### Distribuição Probabilística

A probabilidade de transição não segue uma Normal, mas sim uma **Distribuição Estável de Lévy**:

$$
S_{t+\Delta t} \sim S_t + \text{Deriva} + \text{Estavel}(\alpha, \beta, \gamma, \delta)
$$

Isso permite que o modelo gere cenários de alta volatilidade e saltos abruptos de preço com maior fidelidade à realidade do mercado brasileiro.

## Metodologia de Previsão (2026-2030)

1. **Coleta de Dados**: Dados históricos diários de ativos da B3 (VALE3, PETR4, ITUB4, etc.).
2. **Calibração**:
   - Ajuste de $\alpha$ e $\beta$ utilizando Algoritmo de Máxima Verossimilhança nas séries de retornos logarítmicos.
   - Estimativa de tendência $\mu(t)$ via regressão linear segmentada.
3. **Simulação de Monte Carlo**: Geração de 10.000 caminhos possíveis para cada ativo até 2030, utilizando a EDE de Wilcke.

## Resultados

Os gráficos gerados na pasta `saida/` demonstram as previsões, incluindo intervalos de confiança que capturam a incerteza expandida devido às caudas pesadas.

---
*Desenvolvido por Luiz Tiago Wilcke com assistência de IA Avançada.*
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(conteudo)
    print("README.md gerado com sucesso!")

if __name__ == "__main__":
    gerar_leiaue()
