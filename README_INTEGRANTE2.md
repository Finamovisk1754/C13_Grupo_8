# Integrante 2 — Grupo 8: malhas e controladores PID

Implementação dos itens **4, 5 e 6** do projeto da disciplina de Sistemas Embarcados.
Compatível com os dados (`t`, `Degrau`, `Saida`) e a metodologia de identificação de `Identificacao.py` do repositório do grupo.

## O que foi mantido

- **Item 4:** gráficos de malha aberta e fechada, tempo de subida, acomodação e erro.
- **Item 5:** técnica 4 — **CHR com sobrevalor (20%)**; técnica 5 — **Cohen-Coon**.
- **Item 6:** comparação do tempo de resposta com conclusão gerada dos resultados.
- **Padé:** aproximação de 1ª, 3ª ou 5ª ordem para modelar atraso na função de transferência em malha fechada. O padrão é ordem 5.

## Executar a partir da raiz do GitHub

```bash
python -m pip install -r integrante2/requirements_integrante2.txt
python integrante2/Integrante2_PID.py --dataset Pneumatico_G8.mat --output integrante2/resultados --metodo melhor --pade 5 --setpoint 8
python -m pytest -q integrante2/test_integrante2.py
```

## Arquivos gerados em resultados/

- `01_malha_aberta.png`
- `02_pid_comparativo.png`
- `03_malhas_comparacao.png`
- `metricas_pid.csv`
- `resultado_integrante2.json`
- `analise_desempenho.md`

## Observações de modelagem

Na malha aberta, o gráfico usa a resposta FOPDT analítica com atraso puro; na malha fechada, utiliza-se Padé para aproximar o atraso e simulação de controle linear. O termo Padé não é uma comparação extra, mas parte da implementação. O controlador PID possui derivada filtrada `N=10` para produzir uma função de transferência própria. Não há limites de válvula, saturação nem modelo não linear da pressão; uma eventual resposta inversa no início é artefato do modelo matemático, **não uma pressão física validada**. Os erros finais são calculados no horizonte de simulação.

