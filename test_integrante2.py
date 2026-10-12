from pathlib import Path
import numpy as np
from Integrante2_PID import dados, identificacao, tunings, pade, simulate_pade, run

ROOT=Path(__file__).resolve().parent
DATASET=ROOT.parent/'Pneumatico_G8.mat'

def test_dataset_e_identificacao():
    t,u,y=dados(DATASET)
    models,best,_=identificacao(t,u,y)
    assert len(t)==2001 and best in models
    assert all(m['tau']>0 and m['theta']>0 for m in models.values())

def test_pade_ordens():
    num,den=pade(12,1)
    assert np.allclose(num,[-6,1]) and np.allclose(den,[6,1])
    assert len(pade(12,3)[0])==4
    assert len(pade(12,5)[0])==6

def test_sintonias():
    p=tunings(.114,60,10)
    assert set(p)=={'CHR 20%','Cohen-Coon'}
    assert all(all(float(value)>0 for value in values.values()) for values in p.values())

def test_malha_fechada_e_estabilidade():
    tm=np.arange(2001)*.2
    for pid in tunings(.114,60,10).values():
        curve,stable,poles,met=simulate_pade(.114,60,10,pid,5,tm,1.1,8)
        assert np.isfinite(curve).all() and np.isfinite(poles).all()
        assert isinstance(stable,bool) and met['tempo_subida_s'] is not None

def test_somente_tres_graficos(tmp_path):
    summary=run(DATASET,tmp_path,order=5,setpoint=8)
    assert len(list(tmp_path.glob('*.png')))==3
    assert len(summary['controladores'])==2
    assert (tmp_path/'analise_desempenho.md').exists()
    assert (tmp_path/'metricas_pid.csv').exists()
