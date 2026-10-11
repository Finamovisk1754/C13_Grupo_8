"""
Suposição: o dataset tem colunas de tempo, entrada (degrau) e saída, nessa ordem
(ou com nomes parecidos com t/time, u/input/sp, y/output/pv). Ajuste carregar_dataset()
caso o formato real seja diferente (.mat, .txt, .xlsx etc.).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

SAIDA = Path("resultados")
SAIDA.mkdir(exist_ok=True)


# ----------------------------------------------------------------- dados
def carregar_dataset(caminho):
    if str(caminho).lower().endswith(".mat"):
        import scipy.io as sio
        m = sio.loadmat(caminho, squeeze_me=True)
        return (np.asarray(m["t"], float), np.asarray(m["Degrau"], float),
                np.asarray(m["Saida"], float))
    df = pd.read_csv(caminho, sep=None, engine="python")
    cols = {c.lower().strip(): c for c in df.columns}

    def achar(opcoes, padrao):
        for o in opcoes:
            for k, v in cols.items():
                if o in k:
                    return v
        return df.columns[padrao]

    c_t = achar(["time", "tempo", "t"], 0)
    c_u = achar(["input", "entrada", "sp", "u"], 1)
    c_y = achar(["output", "saida", "saída", "pv", "y"], 2)
    return (df[c_t].to_numpy(float), df[c_u].to_numpy(float), df[c_y].to_numpy(float))


# ----------------------------------------------------------------- modelo
def fopdt_degrau(t, k, tau, theta, du, t0, y0):
    """Resposta analítica de k*e^(-theta s)/(tau s + 1) a um degrau de amplitude du em t0."""
    td = np.clip(t - t0 - theta, 0, None)
    return y0 + k * du * (1 - np.exp(-td / tau))


def eqm(y, yhat):
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def tempo_em(t, ynorm, frac):
    """Primeiro instante em que a resposta normalizada cruza 'frac' (interpolação linear)."""
    idx = np.where(ynorm >= frac)[0]
    if len(idx) == 0:
        raise ValueError(f"Curva não atinge {frac*100:.1f}% do valor final.")
    i = idx[0]
    if i == 0:
        return t[0]
    return np.interp(frac, [ynorm[i - 1], ynorm[i]], [t[i - 1], t[i]])


# ----------------------------------------------------------------- métodos
def smith(t, ynorm, t0):
    t1 = tempo_em(t, ynorm, 0.283) - t0
    t2 = tempo_em(t, ynorm, 0.632) - t0
    tau = 1.5 * (t2 - t1)
    theta = t2 - tau
    return tau, theta, (t1, t2)


def sundaresan(t, ynorm, t0):
    t1 = tempo_em(t, ynorm, 0.353) - t0
    t2 = tempo_em(t, ynorm, 0.853) - t0
    tau = (2 / 3) * (t2 - t1)
    theta = 1.3 * t1 - 0.29 * t2
    return tau, theta, (t1, t2)


# ----------------------------------------------------------------- main
def main(caminho):
    t, u, y = carregar_dataset(caminho)

    # degrau: instante em que a entrada muda
    u0 = np.mean(u[: max(3, len(u) // 50)])
    u_f = np.mean(u[-max(3, len(u) // 20):])
    du = u_f - u0
    t0 = t[np.argmax(np.abs(u - u0) > 0.5 * abs(du))]

    y0 = np.mean(y[t < t0]) if np.any(t < t0) else y[0]
    y_f = np.mean(y[-max(3, len(y) // 20):])
    dy = y_f - y0
    k = dy / du
    # suavização só para detectar os instantes (dataset ruidoso); RMSE usa y bruto
    from scipy.signal import savgol_filter
    ys = savgol_filter(y, 31, 2)
    ynorm = (ys - y0) / dy

    print(f"Degrau: u0={u0:.4g} -> uf={u_f:.4g} (du={du:.4g}) em t0={t0:.4g}")
    print(f"y0={y0:.4g}, yf={y_f:.4g}, dy={dy:.4g}, k={k:.4g}")

    resultados = {}
    for nome, f in (("Smith", smith), ("Sundaresan", sundaresan)):
        tau, theta, (t1, t2) = f(t, ynorm, t0)
        yhat = fopdt_degrau(t, k, tau, theta, du, t0, y0)
        resultados[nome] = dict(k=k, tau=tau, theta=theta, t1=t1, t2=t2,
                                eqm=eqm(y, yhat), yhat=yhat)
        print(f"[{nome}] t1={t1:.4g} t2={t2:.4g} tau={tau:.4g} theta={theta:.4g} "
              f"EQM={resultados[nome]['eqm']:.4g}")

    melhor = min(resultados, key=lambda n: resultados[n]["eqm"])
    print(f"\nMenor EQM: {melhor}")

    # ---- ajuste fino (opcional) partindo do melhor método
    r = resultados[melhor]

    def custo(p):
        kk, tt, th = p
        if tt <= 0 or th < 0:
            return 1e9
        return eqm(y, fopdt_degrau(t, kk, tt, th, du, t0, y0))

    opt = minimize(custo, [r["k"], r["tau"], r["theta"]], method="Nelder-Mead")
    k_f, tau_f, th_f = opt.x
    yfino = fopdt_degrau(t, k_f, tau_f, th_f, du, t0, y0)
    print(f"[Ajuste fino] k={k_f:.4g} tau={tau_f:.4g} theta={th_f:.4g} EQM={eqm(y, yfino):.4g}")

    # ---- gráficos
    plt.figure(figsize=(9, 5))
    plt.plot(t, y, "k", lw=1.5, label="Dataset (original)")
    for nome, r_ in resultados.items():
        plt.plot(t, r_["yhat"], "--", label=f"{nome} (EQM={r_['eqm']:.3g})")
    plt.plot(t, yfino, ":", lw=2, label=f"Ajuste fino (EQM={eqm(y, yfino):.3g})")
    plt.xlabel("Tempo (s)"); plt.ylabel("Saída"); plt.grid(alpha=.3); plt.legend()
    plt.title("Identificação FOPDT - Grupo 8 (cilindro pneumático)")
    plt.savefig(SAIDA / "identificacao_comparacao.png", dpi=300, bbox_inches="tight")

    # FT para uso nas etapas seguintes (Padé) - requer python-control
    try:
        import control as ctrl
        num, den = ctrl.pade(r["theta"], 3)
        G = ctrl.series(ctrl.tf(r["k"], [r["tau"], 1]), ctrl.tf(num, den))
        print("\nFT (Padé 3ª ordem):", G)
    except ImportError:
        print("\n(instale 'control' para gerar a FT com Padé: pip install control)")

    pd.DataFrame([{"metodo": n, "k": v["k"], "tau": v["tau"], "theta": v["theta"],
                   "EQM": v["eqm"]} for n, v in resultados.items()]
                 + [{"metodo": "Ajuste fino", "k": k_f, "tau": tau_f,
                     "theta": th_f, "EQM": eqm(y, yfino)}]
                 ).to_csv(SAIDA / "parametros_identificados.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Pneumatico_G8.mat")