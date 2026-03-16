<div align="center">

# Previsão de Ações do Setor de Energia com LSTM

**Modelo LSTM multivariado para previsão de retorno diário em uma carteira NYSE de energia**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Concluído-1D9E75?style=flat-square)
![Treino](https://img.shields.io/badge/Treino-2010–2019-534AB7?style=flat-square)
![Teste](https://img.shields.io/badge/Teste-2023–2025-534AB7?style=flat-square)

<br>

Treinado com dados de 2010–2019 e avaliado em 2023–2025 com comparação out-of-sample  
contra buy & hold e análise de robustez com **50 seeds independentes**.

</div>

---

## Resultados

> Carteira equal-weight · XOM + CVX + SLB + HAL · Teste 2023–2025

<table>
<tr>
<th align="left">Métrica</th>
<th align="center">LSTM</th>
<th align="center">Buy & Hold</th>
<th align="center">Diferença</th>
</tr>
<tr>
<td><b>Retorno total</b></td>
<td align="center">+12,90%</td>
<td align="center">−9,52%</td>
<td align="center"><b>+22,42pp</b></td>
</tr>
<tr>
<td><b>Sharpe ratio</b></td>
<td align="center"><b>0,236</b></td>
<td align="center">−0,135</td>
<td align="center">+0,371</td>
</tr>
<tr>
<td><b>Max drawdown</b></td>
<td align="center">−20,43%</td>
<td align="center">−33,27%</td>
<td align="center"><b>−12,84pp</b></td>
</tr>
<tr>
<td><b>Capital final ($1.000 invest.)</b></td>
<td align="center"><b>$1.206</b></td>
<td align="center">$922</td>
<td align="center"><b>+$284</b></td>
</tr>
</table>

### Por ativo

| Ticker | LSTM | Buy & Hold | Alpha | |
|:------:|:----:|:----------:|:-----:|:---:|
| **XOM** | −2,44% | +17,59% | −20,03pp | B&H ganhou |
| **CVX** | +41,13% | +1,20% | +39,93pp | ✅ LSTM ganhou |
| **SLB** | +38,58% | −25,32% | +63,89pp | ✅ LSTM ganhou |
| **HAL** | −14,86% | −24,59% | +9,73pp | ✅ LSTM ganhou |

O modelo opera com threshold dinâmico no percentil `p30` de `|ŷ|` — fica em cash 26–39% dos dias de negociação nos sinais de baixa convicção, reduzindo exposição desnecessária ao mercado.

---

## Robustez — 50 Seeds Independentes

O pipeline completo foi retreinado **50 vezes** com inicializações diferentes para verificar que os resultados não dependem de uma semente sortuda.

<div align="center">

### 🏆 LSTM bateu o buy & hold em 45 de 50 runs — 90% de consistência

</div>

| Métrica | Média | Desvio | Mínimo | Máximo |
|:--------|------:|-------:|-------:|-------:|
| Sharpe (carteira) | 0,26 | 0,38 | −0,59 | **1,25** |
| Alpha vs B&H ($) | +$284 | $242 | −$153 | **+$1.061** |
| Capital final LSTM | $1.206 | $242 | $769 | **$1.983** |
| Max drawdown | −0,25 | 0,08 | −0,47 | −0,12 |

> Capital B&H permanece fixo em $922 em todos os runs.  
> Alpha médio de **+$284** por $1.000 investidos, com desvio padrão de $242.

---

## Carteira

A carteira representa uma **cadeia de valor vertical do setor de petróleo**:

| Ticker | Empresa | Subsetor | Papel |
|:------:|:-------:|:--------:|:-----:|
| **XOM** | ExxonMobil | Petróleo integrado | Âncora — benchmark global |
| **CVX** | Chevron | Petróleo integrado | Âncora — benchmark americano |
| **SLB** | Schlumberger | Serviços de exploração | Defasagem de 1–2 trimestres |
| **HAL** | Halliburton | Serviços de completação | Defasagem de 1–2 trimestres |

**Por que essa combinação?** Quando o WTI sobe, XOM e CVX lucram e aumentam o capex de exploração. Esse investimento chega a SLB e HAL com uma defasagem de 1 a 2 trimestres. A LSTM aprende essa cadeia de causalidade ao treinar os 4 ativos em conjunto — capturando tanto os movimentos comuns quanto as divergências temporais entre as integradas e os prestadores de serviço.

---

## Dados e Split Temporal

```
┌──────────────────┬────────────────────┬──────────────────┐
│     TREINO       │     EXCLUÍDO       │      TESTE       │
│   2010 – 2019    │   2020 – 2022      │   2023 – 2025    │
│   ~2.500 dias    │   pandemia         │   ~600 dias      │
└──────────────────┴────────────────────┴──────────────────┘
```

O período **2020–2022 foi excluído intencionalmente**. A pandemia criou um regime de mercado sem precedente — o WTI ficou negativo em abril de 2020, as correlações históricas do setor quebraram e houve intervenções governamentais massivas. Treinar com esse período ensinaria ao modelo padrões que provavelmente não se repetirão.

> **Regra aplicada:** dados nunca são embaralhados. O split é estritamente temporal. O `StandardScaler` é ajustado exclusivamente no treino e aplicado ao teste sem reajuste — qualquer outra abordagem constitui data leakage.

---

## Features — 13 Entradas em 5 Grupos Não Redundantes

A seleção foi guiada por três critérios: cobertura de grupos de informação distintos, ausência de correlação acima de `|r| > 0,90` entre pares, e estacionariedade comprovada pelo teste ADF. Todas as features de preço usam log-return em vez de preço bruto.

| Feature | Grupo | Justificativa |
|:--------|:-----:|:--------------|
| `logret_close`, `logret_open` | Preço | Estacionárias (ADF p < 0,05). `logret_open` captura o gap overnight — surpresas ocorridas fora do pregão, incluindo movimentos noturnos no WTI |
| `volume_ratio` | Volume | Volume normalizado pela média de 20 dias. Detecta anomalias de participação independente do nível absoluto |
| `sma_diff`, `ema_diff` | Tendência | `sma_diff` = (SMA50 − SMA200) / SMA200 — codifica regime bull/bear institucional. `ema_diff` captura tendência de curto prazo com mais reatividade |
| `rsi14`, `macd_hist` | Momentum | RSI mede sobrecompra/sobrevenda em escala 0–100. Histograma MACD mede aceleração da tendência. Os dois têm baixa correlação entre si e são complementares |
| `atr14`, `bb_pct_b` | Volatilidade | ATR normalizado captura o regime de volatilidade, essencial num setor que alterna entre crises e booms. `bb_pct_b` combina tendência e volatilidade numa métrica 0–1 |
| `wti_logret`, `brent_logret`, `spread_wb`, `ng_logret` | Macro | WTI é o driver causal direto. Brent captura choques geopolíticos que precedem o WTI. O spread WTI–Brent reflete gargalos regionais. Gás natural tem ciclo próprio não sincronizado com o petróleo |

<details>
<summary><b>Features eliminadas por redundância (|r| > 0,90)</b></summary>

<br>

| Feature eliminada | Motivo |
|:-----------------|:-------|
| `williams_r` | Correlação > 0,95 com `rsi14` — informação idêntica |
| `stoch_k`, `stoch_d` | Coberto conjuntamente por RSI + MACD |
| `CCI`, `ROC` | Derivados de variáveis já presentes |
| SMA20, SMA50, SMA200 brutos | Substituídos pelo `sma_diff` normalizado |
| `bb_width` | Redundante com `atr14` normalizado |
| `vol_hist` | Redundante com `atr14` |
| `obv_diff` | Coberto pelo `volume_ratio` |

A justificativa completa com testes ADF, matrizes de correlação e scores de Mutual Information está em `exploratory_analysis.ipynb`.

</details>

---

## Arquitetura

```
Entrada           LSTM × 2 camadas           Saída
(batch, 20, 13) ───────────────────────► Linear(128 → 1)
                  hidden = 128              sem ativação
                  dropout = 0.2
```

**Por que sem ativação na saída?** Log-returns são valores reais contínuos — podem ser +0,03 ou −0,05. Funções como `sigmoid` ou `tanh` restringiriam a saída a intervalos fixos, impedindo o modelo de prever retornos de maior magnitude. A camada `Linear` pura é a escolha correta para regressão de valores ilimitados.

### Função de Perda Híbrida

```
Loss = RMSE + λ · (1 − Sharpe normalizado)       λ = 0,3
```

RMSE puro minimiza o erro de previsão mas pode gerar péssimos sinais de trading — um modelo pode ter baixo RMSE e errar sistematicamente a *direção* dos movimentos, que é o que determina o resultado financeiro. O termo Sharpe penaliza estratégias com baixo retorno ajustado ao risco.

### Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|:----------|:-----:|:-------------|
| Lookback (janela) | 20 dias | 1 mês de pregão — cobre RSI/SMA20 aquecidos, unidade natural de rebalanceamento institucional |
| Hidden size | 128 | Capacidade suficiente para 13 features × 4 ativos |
| Num layers | 2 | Captura padrões temporais em múltiplas escalas |
| Dropout | 0,2 | Regularização nas conexões entre camadas LSTM |
| Otimizador | Adam, lr = 1e-3 | Padrão para séries financeiras |
| Scheduler | ReduceLROnPlateau (patience=5, factor=0,5) | Reduz LR quando a loss estagna |
| Early stopping | patience = 10, máx 100 épocas | Interrompe antes de overfitting |
| Gradient clipping | max_norm = 1,0 | Previne gradientes explodindo — problema clássico em LSTMs com séries voláteis |
| Batch size | 64 | |
| λ (peso do Sharpe) | 0,3 | |

---

## Threshold Dinâmico

Em vez de um valor fixo e arbitrário, cada ativo recebe seu próprio threshold calculado como o **percentil p30 da distribuição de `|ŷ|` no conjunto de treino**. Isso garante que o filtro seja proporcional à escala das previsões de cada ativo.

```
|ŷ| > threshold  e  ŷ > 0   →   long   (compra)
|ŷ| > threshold  e  ŷ < 0   →   short  (vende a descoberto)
|ŷ| ≤ threshold              →   cash   (protege capital)
```

| Ativo | Threshold calibrado | Dias operando | Dias em cash |
|:-----:|:-------------------:|:-------------:|:------------:|
| XOM | 0,005016 | 73,4% | 26,6% |
| CVX | 0,005788 | 66,0% | 34,0% |
| SLB | 0,006711 | 65,6% | 34,4% |
| HAL | 0,006546 | 61,2% | 38,8% |

O percentil p30 foi selecionado por maximização do Sharpe na distribuição de treino e aplicado **sem reajuste** no teste. Percentis abaixo de p30 colapsam para thresholds numericamente insignificantes porque a distribuição de `|ŷ|` é muito concentrada em torno de zero.

---

## Métricas de Avaliação

### Estatísticas — qualidade da previsão

| Métrica | Fórmula | Interpretação |
|:--------|:-------:|:--------------|
| RMSE | `√mean((ŷ − y)²)` | Penaliza erros grandes mais que pequenos |
| MAE | `mean(\|ŷ − y\|)` | Complementa o RMSE — menos sensível a outliers |
| R² | `1 − SS_res / SS_tot` | Em finanças, R² de 0,02–0,05 já é relevante |
| Acurácia direcional | `mean(sign(ŷ) == sign(y))` | Acima de 52–53% é economicamente significativo |

### Financeiras — qualidade do sinal de trading

| Métrica | Interpretação |
|:--------|:--------------|
| Sharpe anualizado | `(mean(r) / std(r)) × √252`. Sharpe > 1,0 é bom |
| Máximo drawdown | Maior queda acumulada desde o pico até o vale |
| Alpha vs B&H | Diferença de retorno total entre LSTM e buy & hold simples |

---

## Stack

```bash
pip install torch yfinance pandas-ta scikit-learn seaborn matplotlib
```

| Biblioteca | Versão | Uso |
|:----------|:------:|:----|
| `torch` | ≥ 2.0 | Definição da LSTM, loop de treino, loss híbrida |
| `yfinance` | ≥ 0.2 | Download de preços históricos e futuros (WTI, Brent, gás natural) |
| `pandas-ta` | any | Indicadores técnicos. O helper `get_col(df, prefix)` detecta nomes de colunas automaticamente — evita `KeyError` por diferenças de versão |
| `scikit-learn` | ≥ 1.3 | `StandardScaler`, `mean_squared_error`, `r2_score` |
| `matplotlib` / `seaborn` | — | Curvas de capital, scatter de previsões, heatmaps |

---

## Notebooks

### [`exploratory_analysis.ipynb`](exploratory_analysis.ipynb)

Justificativa empírica de todas as decisões de feature engineering. Nenhuma feature foi escolhida sem evidência quantitativa.

| Análise | O que prova |
|:--------|:-----------|
| **Teste ADF** | Log-return é estacionário (p < 0,05); preço bruto não é — justifica a transformação |
| **Correlação — candidatas** | Identifica todos os pares com \|r\| > 0,90, revelando as redundâncias a eliminar |
| **Correlação — selecionadas** | Confirma ausência de redundância no conjunto final de 13 features |
| **Mutual Information** | Mede o poder preditivo não-linear de cada feature sobre o retorno do dia seguinte |
| **Correlação cruzada** | Justifica a arquitetura multivariada: 4 ativos correlacionados mas com divergências temporais exploráveis |

### [`energy_lstm.ipynb`](energy_lstm.ipynb)

Pipeline completo de ponta a ponta:

| # | Seção |
|:-:|:------|
| 1 | Imports e configuração |
| 2 | Coleta de dados via `yfinance` |
| 3 | Engenharia de features com `pandas-ta` |
| 4 | Pipeline — janelas deslizantes, normalização, split |
| 5 | Arquitetura `EnergyLSTM` |
| 6 | Loop de treino com early stopping e gradient clipping |
| 7 | Threshold dinâmico — calibração por percentil no treino |
| 8 | Avaliação estatística no teste |
| 9 | Estratégia de trading com threshold |
| 10 | Comparação com buy & hold — curvas de capital e drawdown |
| 11 | Análise final por ativo e carteira |
| 12 | Simulação de $1.000 investidos |
| 13 | Análise de robustez — 50 seeds com CSV incremental |

---

## Reprodutibilidade

O CSV com os resultados dos 50 seeds é salvo após **cada run individualmente** em `robustez_runs.csv`. Se a execução for interrompida, o notebook detecta o arquivo existente e retoma do seed seguinte automaticamente.

```
[ 1/50]  seed= 0  sharpe=-0.588  alpha=$-146   capital=$776    epochs=100  51s  ✗
[ 2/50]  seed= 1  sharpe=+0.405  alpha=$+344   capital=$1,266  epochs=100  51s  ✓
[11/50]  seed=10  sharpe=+1.249  alpha=$+1,061 capital=$1,983  epochs=100  51s  ✓
...
[50/50]  seed=49  sharpe=+0.522  alpha=$+508   capital=$1,430  epochs=100  51s  ✓
```

Para reproduzir os resultados com `seed=42`:

```bash
jupyter nbconvert --to notebook --execute exploratory_analysis.ipynb --output exploratory_analysis_executed.ipynb
jupyter nbconvert --to notebook --execute energy_lstm.ipynb --output energy_lstm_executed.ipynb
```

---

## Limitações

| Limitação | Impacto |
|:----------|:--------|
| **Custos de transação não modelados** | Bid-ask spread (~0,05%) e comissões reduziriam o alpha realizado. Uma estratégia ativa diária acumula custos relevantes |
| **Short selling sem custo de aluguel** | A estratégia assume posições vendidas sem modelar o custo de aluguel de ações |
| **XOM com alpha negativo** | Hipótese: crack spread (margem de refino) é um driver relevante para integradas e não está incluído como feature |
| **Período de teste** | 2023–2025 coincide com recuperação pós-pandemia de SLB/HAL — walk-forward em múltiplos períodos fortaleceria a análise |

---

## Trabalho Futuro

- [ ] Adicionar **crack spread** (WTI − gasolina) como feature macro para as integradas
- [ ] Incorporar **sentiment de notícias** via NLP em headlines do setor de energia
- [ ] Comparar com arquiteturas **Temporal Fusion Transformer (TFT)** e **N-BEATS**
- [ ] Implementar **walk-forward validation** com janelas deslizantes de treino/teste
- [ ] Modelar **custos de transação** explicitamente na função de perda e na simulação

---

## Referências

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation, 9(8), 1735–1780.
- Fischer, T. & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.
- Murphy, J. J. (1999). *Technical Analysis of the Financial Markets*. New York Institute of Finance.
- de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
