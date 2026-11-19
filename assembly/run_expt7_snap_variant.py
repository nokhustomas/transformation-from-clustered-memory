import sys, torch, time
import torch.nn.functional as F
sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../experiments')
from models import FullAttentionLM
from concept_lm_data import make_concept_spec, generate_batch
from registry import AssemblyRegistry
from run_expt2_completion import DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED, OUT_DIR
from run_expt3_stress import EMIT_EXTREME
import os

spec = make_concept_spec(8, 8, emit_probs=EMIT_EXTREME)
torch.manual_seed(0)
model = FullAttentionLM(64, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'expt3_model_seed0.pt')))
model.eval()
st = torch.load(os.path.join(OUT_DIR, 'expt3_registries_seed0.pt'))
reg = AssemblyRegistry(**st['L1']['config'])
for key in ('freq','act_ema','active','count','created_at'): setattr(reg, key, st['L1'][key])

acts = {}
hcap = model.layers[1].ffn[1].register_forward_hook(lambda m,i,o: acts.__setitem__(1, o.detach()))

emb_typ = torch.zeros(reg.K, DIM); emb_cnt = torch.zeros(reg.K)
g = torch.Generator().manual_seed(DATA_SEED + 7777)
with torch.no_grad():
    for _ in range(120):
        ids, _ = generate_batch(16, SEQ_LEN, spec, generator=g)
        emb = model.tok_emb(ids).reshape(-1, DIM)
        model(ids)
        a, _ = reg.assign(reg.extract_sets(acts[1].reshape(-1, DIM*4)))
        for k in a[a>=0].unique().tolist():
            sel = a==k; n = int(sel.sum()); w = n/(emb_cnt[k]+n)
            emb_typ[k] = (1-w)*emb_typ[k] + w*emb[sel].mean(0); emb_cnt[k] += n

sub = {'cur': None}
hemb = model.tok_emb.register_forward_hook(lambda m,i,o: sub['cur'])
stream = torch.Generator().manual_seed(DATA_SEED + 555)
res = {'off': {'rare': [], 'freq': []}, 'snap': {'rare': [], 'freq': []}}
t0 = time.time()
with torch.no_grad():
    for bi in range(1500):
        ids, _ = generate_batch(32, SEQ_LEN, spec, generator=stream)
        B, T = ids.shape; rk = ids % 8
        sub['cur'] = None
        logits = model(ids)
        lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').view(B, -1)
        rmask, fmask = rk[:, :-1] >= 6, rk[:, :-1] < 2
        res['off']['rare'].append(lp[rmask]); res['off']['freq'].append(lp[fmask])
        a, rho = reg.assign(reg.extract_sets(acts[1].reshape(-1, DIM*4)))
        hit = (a>=0) & (emb_cnt[a.clamp_min(0)]>0)
        emb0 = model.tok_emb.weight[ids.reshape(-1)]  # 原始嵌入(绕开hook直接查表)
        typ = emb_typ[a.clamp_min(0)]
        emb2 = torch.where(hit.unsqueeze(-1), typ, emb0)  # 全量替换
        sub['cur'] = emb2.view(B, T, DIM)
        logits2 = model(ids)
        lp2 = F.cross_entropy(logits2[:, :-1].reshape(-1, logits2.shape[-1]), ids[:, 1:].reshape(-1), reduction='none').view(B, -1)
        res['snap']['rare'].append(lp2[rmask]); res['snap']['freq'].append(lp2[fmask])
        sub['cur'] = None
hemb.remove(); hcap.remove()
for b in ('rare','freq'):
    ref = torch.cat(res['off'][b]); x = torch.cat(res['snap'][b]); d = x - ref
    print(f'{b}: n={ref.shape[0]} off={float(ref.mean()):.4f} snap Δ={float(d.mean()):+.4f}(±{float(d.std()/d.shape[0]**0.5):.4f})')
print(f'({time.time()-t0:.0f}s)  参照: oracle rare Δ=-0.0786±0.0166')
