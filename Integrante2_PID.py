"""Integrante 2 - Grupo 8: malhas aberta/fechada, CHR 20% e Cohen-Coon.
Modelo linear FOPDT com atraso aproximado por Padé.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter, TransferFunction, step
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def dados(path):
    m = loadmat(path, squeeze_me=True)
    t,u,y = (np.asarray(m[k], dtype=float).ravel() for k in ('t','Degrau','Saida'))
    if len(t)<50 or not(len(t)==len(u)==len(y)) or not np.all(np.diff(t)>0) or not np.isfinite(np.concatenate([t,u,y])).all():
        raise ValueError('Dataset inválido: t, Degrau e Saida devem ter amostras finitas e tempo crescente')
    return t,u,y


def identificacao(t,u,y):
    """Mesma metodologia e convenção do Identificacao.py original do GitHub."""
    u0=float(np.mean(u[:max(3,len(u)//50)]));uf=float(np.mean(u[-max(3,len(u)//20):]));du=uf-u0
    if abs(du)<1e-12: raise ValueError('Degrau de entrada nulo')
    mask=np.flatnonzero(np.abs(u-u0)>.5*abs(du))
    if len(mask)==0: raise ValueError('Degrau de entrada não encontrado')
    t0=float(t[mask[0]]); y0=float(np.mean(y[t<t0])) if np.any(t<t0) else float(y[0])
    yf=float(np.mean(y[-max(3,len(y)//20):]));dy=yf-y0
    if abs(dy)<1e-12: raise ValueError('Variação de saída nula')
    smooth=savgol_filter(y,31,2); z=(smooth-y0)/dy
    def cross(fr):
        ix=np.flatnonzero(z>=fr)
        if len(ix)==0: raise ValueError(f'Curva não cruza {fr:.3f}')
        i=int(ix[0]);return float(t[0] if i==0 else np.interp(fr,[z[i-1],z[i]],[t[i-1],t[i]]))-t0
    a,b,c,d=cross(.283),cross(.632),cross(.353),cross(.853)
    options={'Smith':(1.5*(b-a),b-1.5*(b-a)), 'Sundaresan':((2/3)*(d-c),1.3*c-.29*d)}
    k=dy/du;result={}
    for name,(tau,theta) in options.items():
        if tau<=0 or theta<0: raise ValueError(f'Parâmetros não físicos para {name}')
        pred=y0+k*du*(1-np.exp(-np.maximum(t-t0-theta,0)/tau))
        result[name]={'k':float(k),'tau':float(tau),'theta':float(theta),'rmse':float(np.sqrt(np.mean((pred-y)**2)))}
    return result,min(result,key=lambda n:result[n]['rmse']),{'t0':t0,'u0':u0,'du':du,'y0':y0,'yf':yf,'uf':uf}


def tunings(k,tau,theta):
    if min(k,tau,theta)<=0: raise ValueError('Sintonia requer k, tau, theta positivos')
    r=theta/tau
    return {'CHR 20%':{'Kp':.95*tau/(k*theta),'Ti':1.357*tau,'Td':.473*theta},
            'Cohen-Coon':{'Kp':tau/(k*theta)*(16+3*r)/12,'Ti':theta*(32+6*r)/(13+8*r),'Td':theta*4/(11+2*r)}}


def pade(theta,order):
    if order not in (1,3,5): raise ValueError('Ordem de Padé deve ser 1, 3 ou 5')
    coef=[math.factorial(2*order-j)*math.factorial(order)/(math.factorial(2*order)*math.factorial(j)*math.factorial(order-j))*theta**j for j in range(order+1)]
    return np.array([(-1)**j*coef[j] for j in range(order,-1,-1)]),np.array([coef[j] for j in range(order,-1,-1)])


def pid_tf(pid, filter_n=10):
    """C(s)=Kp(1+1/(Ti s)+Td s/(1+Tf s)); Tf=Td/N."""
    kp,ti,td=pid['Kp'],pid['Ti'],pid['Td'];tf=td/filter_n
    numerator=kp*np.array([ti*(tf+td), ti+tf, 1.])
    denominator=np.array([ti*tf, ti, 0.])
    return numerator,denominator


def pade_closed_loop(k,tau,theta,pid,order):
    pn,pd=pade(theta,order)
    cn,cd=pid_tf(pid)
    plant_n=k*pn;plant_d=np.polymul([tau,1.],pd)
    ln=np.polymul(cn,plant_n);ld=np.polymul(cd,plant_d)
    return ln,np.polyadd(ld,ln)


def metrics(time,curve,y0,target):
    amp=target-y0
    if abs(amp)<1e-12: raise ValueError('Setpoint precisa diferir da pressão inicial')
    z=(np.asarray(curve)-y0)/amp
    def crossing(level):
        ix=np.flatnonzero(z>=level)
        if not len(ix): return None
        j=int(ix[0]);return float(time[0] if j==0 else np.interp(level,[z[j-1],z[j]],[time[j-1],time[j]]))
    a,b=crossing(.1),crossing(.9)
    outside=np.flatnonzero(np.abs(np.asarray(curve)-target)>.02*abs(amp))
    ts=(float(time[outside[-1]+1]) if len(outside) and outside[-1]+1<len(time) else
        (float(time[0]) if not len(outside) else None))
    return {'tempo_subida_s':None if a is None or b is None else round(b-a,3),
            'tempo_acomodacao_s':None if ts is None else round(ts,3),
            'overshoot_pct':round(max(0.,float(z.max()-1)*100),3),
            'erro_final_bar':round(float(target-curve[-1]),5),
            'pressao_minima_bar':round(float(np.min(curve)),4),
            'pressao_inicial_bar':round(float(curve[0]),4)}


def simulate_pade(k,tau,theta,pid,order,time,y0,target):
    numerator,denominator=pade_closed_loop(k,tau,theta,pid,order)
    poles=np.roots(denominator); stable=bool(np.all(np.real(poles)<-1e-9))
    _,r=step(TransferFunction(numerator,denominator),T=time)
    curve=y0+(target-y0)*r
    return curve,stable,poles,metrics(time,curve,y0,target)


def plot(time,series,title,path,setpoint=None):
    fig,ax=plt.subplots(figsize=(11,5.5))
    for name,values in series.items(): ax.plot(time,values,label=name,lw=1.7)
    if setpoint is not None: ax.axhline(setpoint,color='black',ls='--',lw=1,label='SetPoint')
    ax.set(xlabel='Tempo desde o degrau (s)',ylabel='Pressão (bar)',title=title)
    ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(path,dpi=165);plt.close(fig)


def run(dataset, output, metodo='melhor', order=5, setpoint=None, duration=400):
    """Executa apenas os itens 4, 5 e 6 do enunciado; Padé representa o atraso."""
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    t,u,y=dados(dataset)
    identified,best,base=identificacao(t,u,y)
    selected=best if metodo=='melhor' else metodo
    if selected not in identified: raise ValueError('Método de identificação inválido')
    param=identified[selected]
    k,tau,theta=[param[n] for n in ('k','tau','theta')]
    y0=base['y0']; target=base['yf'] if setpoint is None else float(setpoint)
    dt=float(np.median(np.diff(t)))
    if duration<=0 or dt<=0: raise ValueError('Horizonte inválido')
    time=np.arange(round(duration/dt)+1)*dt
    # Malha aberta em termos absolutos: degrau equivalente para mesma saída final.
    du=(target-y0)/k
    open_curve=y0+k*du*(1-np.exp(-np.maximum(time-theta,0)/tau))
    open_metrics=metrics(time,open_curve,y0,target)
    plot(time, {'Malha aberta (FOPDT)':open_curve},
         'Resposta da planta em malha aberta', output/'01_malha_aberta.png',target)
    controllers={}; responses={}
    for name,pid in tunings(k,tau,theta).items():
        curve,stable,poles,met=simulate_pade(k,tau,theta,pid,order,time,y0,target)
        responses[name]=curve
        controllers[name]={
            'parametros':{key:round(float(value),6) for key,value in pid.items()},
            'estavel_linear':stable,
            'polo_real_max':round(float(max(np.real(poles))),8),
            'metricas':met,
        }
    plot(time,responses,'Sintonia PID: CHR com sobrevalor × Cohen-Coon',
         output/'02_pid_comparativo.png',target)
    plot(time,{'Malha aberta (FOPDT)':open_curve,**responses},
         'Malha aberta × malha fechada: mesmo valor inicial e referência final',
         output/'03_malhas_comparacao.png',target)
    summary={
      'dataset':Path(dataset).name,'identificacao':identified,
      'melhor_identificacao':best,'modelo_escolhido':selected,
      'variaveis_ensaio':base,'pade_ordem':order,'setpoint_bar':target,
      'malha_aberta_equivalente':open_metrics,'controladores':controllers,
      'modelo':'Planta FOPDT. Malha aberta com atraso puro analítico; malha fechada linear com atraso por Padé e PID com filtro derivativo N=10.',
      'observacao':'A aproximação Padé e o PID linear não impõem limites físicos de pressão ou atuador. Respostas inversas iniciais são artefatos possíveis do modelo racional.',
    }
    (output/'resultado_integrante2.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (output/'metricas_pid.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['metodo','Kp','Ti','Td','estavel_linear',*open_metrics.keys()]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,item in controllers.items():
            writer.writerow({'metodo':name,**item['parametros'],
                             'estavel_linear':item['estavel_linear'],**item['metricas']})
    def winner(key):
        eligible=[(name,entry['metricas'][key]) for name,entry in controllers.items() if entry['metricas'][key] is not None]
        return min(eligible,key=lambda p:p[1])[0] if eligible else 'Indeterminado'
    doc=['# Integrante 2 — Grupo 8: resultados dos itens 4, 5 e 6','',
      f'Dataset: `{Path(dataset).name}`. Modelo identificado: **{selected}** (menor RMSE: {best}).',
      f'Modelo FOPDT: k={k:.6f}, τ={tau:.3f} s, θ={theta:.3f} s.',
      f'Condições: pressão inicial={y0:.4f} bar; SetPoint={target:.4f} bar; horizonte={duration:g} s; Padé ordem {order}.','',
      '## Item 4 — Malha aberta e fechada','',
      'A malha aberta recebe um degrau equivalente para atingir o mesmo valor final de referência, e a malha fechada usa o SetPoint. As entradas não são fisicamente idênticas.',
      'Tempo de subida medido entre 10–90% e acomodação na banda de ±2% do incremento.', '',
      '| Indicador | Aberta | CHR 20% | Cohen-Coon |','|---|---:|---:|---:|']
    for key,label in [('tempo_subida_s','Subida (s)'),('tempo_acomodacao_s','Acomodação (s)'),('overshoot_pct','Overshoot (%)'),('erro_final_bar','Erro na última amostra (bar)')]:
        doc.append(f'| {label} | {open_metrics[key]} | {controllers["CHR 20%"]["metricas"][key]} | {controllers["Cohen-Coon"]["metricas"][key]} |')
    doc.extend(['', '## Itens 5 e 6 — Sintonias e desempenho', '',
        '| Técnica | Kp | Ti (s) | Td (s) | Estável (modelo linear) |',
        '|---|---:|---:|---:|---|'])
    for name,item in controllers.items():
        p=item['parametros'];doc.append(f'| {name} | {p["Kp"]:.4f} | {p["Ti"]:.4f} | {p["Td"]:.4f} | {"Sim" if item["estavel_linear"] else "Não"} |')
    doc.extend(['',f'- Menor tempo de subida: **{winner("tempo_subida_s")}**.',
        f'- Menor tempo de acomodação: **{winner("tempo_acomodacao_s")}**.',
        '- Para concluir sobre “menor tempo de resposta”, consideramos prioritariamente o tempo de acomodação (atingir e manter a referência).',
        '- A técnica CHR 20% é uma regra nominal; o percentual não é garantia de overshoot observado.',
        '', '## Hipóteses e limites',
        '- O atraso da planta é aproximado por Padé na malha fechada; a malha aberta apresenta a resposta FOPDT analítica.',
        '- Foi utilizado PID com derivada filtrada (N=10) para obter função de transferência própria; os ganhos Kp, Ti e Td seguem as fórmulas do enunciado.',
        '- Não há saturação, anti-windup, dinâmica de válvula ou restrição física de pressão. Valores iniciais negativos no modelo aproximado não são medições reais.',
        '- O erro final é medido no fim do horizonte, não é cálculo analítico do erro em regime infinito.',
        '', '## Figuras obrigatórias',
        '1. `01_malha_aberta.png`', '2. `02_pid_comparativo.png`', '3. `03_malhas_comparacao.png`',''])
    (output/'analise_desempenho.md').write_text('\n'.join(doc),encoding='utf-8')
    return summary


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',default='Pneumatico_G8.mat')
    a.add_argument('--output',default='resultados')
    a.add_argument('--metodo',choices=['melhor','Smith','Sundaresan'],default='melhor')
    a.add_argument('--pade',type=int,choices=[1,3,5],default=5)
    a.add_argument('--setpoint',type=float)
    a.add_argument('--duracao',type=float,default=400)
    p=a.parse_args()
    res=run(p.dataset,p.output,p.metodo,p.pade,p.setpoint,p.duracao)
    print(json.dumps({'modelo':res['modelo_escolhido'],
        'malha_aberta':res['malha_aberta_equivalente'],
        'controladores':res['controladores']},ensure_ascii=False,indent=2))
