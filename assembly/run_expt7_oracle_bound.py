import sys, torch
import torch.nn.functional as F
sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../experiments')
from models import FullAttentionLM
from concept_lm_data import make_concept_spec, generate_batch
from run_expt3_stress import EMIT_EXTREME
from run_expt2_completion import DATA_SEED

spec = make_concept_spec(8, 8, emit_probs=EMIT_EXTREME)
torch.manual_seed(0)
m = FullAttentionLM(64, 64, 2, 4, max_len=64)
m.load_state_dict(torch.load('results/expt3_model_seed0.pt'))
m.eval()
cls = torch.arange(64) // 8
rank = torch.arange(64) % 8
E = m.tok_emb.weight.detach().clone()

E_cent = E.clone(); E_ex = E.clone()
for t in range(64):
    if rank[t] >= 6:
        donors = (cls == cls[t]) & (rank < 4)
        E_cent[t] = E[donors].mean(0)
        E_ex[t] = E[((cls == cls[t]) & (rank == 0)).nonzero().item()]

N_BATCHES = 1500
def collect(emb_weight):
    with torch.no_grad(): m.tok_emb.weight.copy_(emb_weight)
    g = torch.Generator().manual_seed(DATA_SEED + 555)
    rare, freq = [], []
    with torch.no_grad():
        for _ in range(N_BATCHES):
            ids, _ = generate_batch(32, 64, spec, generator=g)
            logits = m(ids)
            lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').view(32, -1)
            rk = ids[:, :-1] % 8
            rare.append(lp[rk >= 6]); freq.append(lp[rk < 2])
    return torch.cat(rare), torch.cat(freq)

r0, f0 = collect(E)
rc, fc = collect(E_cent)
re, fe = collect(E_ex)
with torch.no_grad(): m.tok_emb.weight.copy_(E)

def mse(x): return float(x.mean()), float(x.std()/x.shape[0]**0.5)
rm, rse = mse(r0); fm, fse = mse(f0)
gap = rm - fm; gap_se = (rse**2 + fse**2) ** 0.5
print(f'n_rare={r0.shape[0]}, n_freq={f0.shape[0]}')
print(f'base:   rare={rm:.4f}±{rse:.4f}  freq={fm:.4f}±{fse:.4f}  缺口={gap:+.4f}±{gap_se:.4f}')
dc = rc - r0; de = re - r0
print(f'质心oracle: rare={mse(rc)[0]:.4f}  配对Δ={float(dc.mean()):+.4f}±{float(dc.std()/dc.shape[0]**0.5):.4f}')
print(f'样本oracle: rare={mse(re)[0]:.4f}  配对Δ={float(de.mean()):+.4f}±{float(de.std()/de.shape[0]**0.5):.4f}')
res_gap_c = mse(rc)[0] - mse(fc)[0]; res_gap_e = mse(re)[0] - mse(fe)[0]
print(f'oracle后残余缺口: 质心 {res_gap_c:+.4f}  样本 {res_gap_e:+.4f}')
