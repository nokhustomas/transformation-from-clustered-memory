import sys, torch, os, json
sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../experiments')
from models import FullAttentionLM
from registry import AssemblyRegistry
from run_expt6_realtext import load_corpus, make_batch, DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED
from run_expt10_detector import auc, K_BIG, TAU_BIG, AF, FLR, N_SEQ
from run_expt11_harden import FaultInjector

ids_corpus, stoi, itos = load_corpus()
V = max(itos.keys())+1
torch.manual_seed(0)
model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
model.load_state_dict(torch.load('results/expt6_model_seed0.pt'))
model.eval()

inj = FaultInjector(model.layers[1].ffn[1])
caps = {}
h = model.layers[1].ffn[1].register_forward_hook(lambda m,i,o: caps.__setitem__(1, o.detach()))

reg = AssemblyRegistry(DIM*4, K_BIG, TAU_BIG, FLR, act_frac=AF)
g = torch.Generator().manual_seed(DATA_SEED+8888)
with torch.no_grad():
    for _ in range(150):
        ids = make_batch(ids_corpus, 16, SEQ_LEN, g)
        model(ids)
        flat = caps[1].reshape(-1, DIM*4)[::8]
        reg.register(reg.extract_sets(flat), values=flat)

def scores(batches, kind='none', param=0.0, stuck_frac=None):
    s_set, s_val, s_comb = [], [], []
    inj.kind = kind; inj.param = param
    if kind=='stuck': inj.set_stuck(stuck_frac, DIM*4)
    for bi, ids in enumerate(batches):
        inj.batch_idx = bi+1
        with torch.no_grad(): model(ids)
        flat = caps[1].reshape(-1, DIM*4)
        sets = reg.extract_sets(flat)
        a, rho = reg.assign(sets)
        # 值失配: 命中组的成员 unit 上, 实际激活 vs 组典型激活 的相对差
        k = a.clamp_min(0)
        mem = reg.membership[k]          # [P,U]
        typ = reg.act_ema[k]
        hit = (a>=0).unsqueeze(-1) & mem
        diff = (flat - typ).abs() * hit.float()
        denom = (typ.abs() * hit.float()).sum(-1).clamp_min(1e-6)
        vmis = diff.sum(-1) / denom      # 每位置的相对值失配(未命中=0)
        B = ids.shape[0]
        r = rho.clamp_min(0.0).view(B,-1).mean(1)
        v = vmis.view(B,-1).mean(1)
        s_set += (-r).tolist(); s_val += v.tolist(); s_comb += (v - r).tolist()
    inj.kind='none'
    return s_set, s_val, s_comb

eg = torch.Generator().manual_seed(DATA_SEED+424242)
batches = [make_batch(ids_corpus, 16, SEQ_LEN, eg) for _ in range(N_SEQ//16)]
n_set, n_val, n_comb = scores(batches)

out = {}
for name, kw in (('quant8级', dict(kind='quant', param=8)), ('quant4级', dict(kind='quant', param=4)),
                 ('noise0.5x', dict(kind='noise', param=0.5)), ('stuck10%', dict(kind='stuck', param=0, stuck_frac=0.1)),
                 ('drop30', dict(kind='drop', param=0.3))):
    f_set, f_val, f_comb = scores(batches, **kw)
    a1,se1 = auc(f_set, n_set); a2,se2 = auc(f_val, n_val); a3,se3 = auc(f_comb, n_comb)
    out[name] = dict(set=(a1,se1), val=(a2,se2), comb=(a3,se3))
    print(f'{name:10s}: 开关读数={a1:.3f}±{se1:.3f} | 亮度读数={a2:.3f}±{se2:.3f} | 合并={a3:.3f}±{se3:.3f}', flush=True)
json.dump({k:{kk:list(vv) for kk,vv in v.items()} for k,v in out.items()}, open('results/expt11b_valuescore.json','w'))
print('完成')
