# Integrante 2 — Grupo 8: resultados dos itens 4, 5 e 6

Dataset: `Pneumatico_G8.mat`. Modelo identificado: **Smith** (menor RMSE: Smith).
Modelo FOPDT: k=0.114711, τ=54.738 s, θ=13.935 s.
Condições: pressão inicial=1.1143 bar; SetPoint=8.0000 bar; horizonte=400 s; Padé ordem 5.

## Item 4 — Malha aberta e fechada

A malha aberta recebe um degrau equivalente para atingir o mesmo valor final de referência, e a malha fechada usa o SetPoint. As entradas não são fisicamente idênticas.
Tempo de subida medido entre 10–90% e acomodação na banda de ±2% do incremento.

| Indicador | Aberta | CHR 20% | Cohen-Coon |
|---|---:|---:|---:|
| Subida (s) | 120.273 | 20.622 | 17.452 |
| Acomodação (s) | 228.2 | 94.4 | 163.2 |
| Overshoot (%) | 0.0 | 20.491 | 78.236 |
| Erro na última amostra (bar) | 0.00596 | 0.00184 | 0.0003 |

## Itens 5 e 6 — Sintonias e desempenho

| Técnica | Kp | Ti (s) | Td (s) | Estável (modelo linear) |
|---|---:|---:|---:|---|
| CHR 20% | 32.5326 | 74.2801 | 6.5910 | Sim |
| Cohen-Coon | 47.8391 | 31.0703 | 4.8430 | Sim |

- Menor tempo de subida: **Cohen-Coon**.
- Menor tempo de acomodação: **CHR 20%**.
- Para concluir sobre “menor tempo de resposta”, consideramos prioritariamente o tempo de acomodação (atingir e manter a referência).
- A técnica CHR 20% é uma regra nominal; o percentual não é garantia de overshoot observado.

## Hipóteses e limites
- O atraso da planta é aproximado por Padé na malha fechada; a malha aberta apresenta a resposta FOPDT analítica.
- Foi utilizado PID com derivada filtrada (N=10) para obter função de transferência própria; os ganhos Kp, Ti e Td seguem as fórmulas do enunciado.
- Não há saturação, anti-windup, dinâmica de válvula ou restrição física de pressão. Valores iniciais negativos no modelo aproximado não são medições reais.
- O erro final é medido no fim do horizonte, não é cálculo analítico do erro em regime infinito.

## Figuras obrigatórias
1. `01_malha_aberta.png`
2. `02_pid_comparativo.png`
3. `03_malhas_comparacao.png`
