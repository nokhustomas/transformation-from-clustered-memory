import sys, torch, os, json
import torch.nn.functional as F
sys.path.insert(0,'.'); sys.path.insert(0,'..'); sys.path.insert(0,'../experiments')
from models import FullAttentionLM
from registry import AssemblyRegistry
from run_expt6_realtext import load_corpus, make_batch, DIM, LAYERS, HEADS, SEQ_LEN, DATA_SEED
from run_expt10_detector import auc, K_BIG, TAU_BIG, AF, FLR, N_SEQ, CORRUPT_SEED, OUT_DIR

ids_corpus, stoi, itos = load_corpus()
V = max(itos.keys()) + 1
torch.manual_seed(0)
model = FullAttentionLM(V, DIM, LAYERS, HEADS, max_len=SEQ_LEN)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, 'expt6_model_seed0.pt')))
model.eval()
upper_map = torch.arange(V)
for c, i in stoi.items():
    cu = c.upper()
    if cu != c and cu in stoi: upper_map[i] = stoi[cu]

fault = {'drop': 0.0, 'seed': 0}
def fault_hook(_m,_i,out):
    if fault['drop'] <= 0: return None
    gg = torch.Generator().manual_seed(CORRUPT_SEED + fault['seed'])
    B,T,U = out.shape
    flat = out.reshape(-1,U)
    a = flat.clamp_min(0.0); mx = a.max(-1,keepdim=True).values
    sets_pre = (a > 0.5*mx) & (mx>0)
    drop = (torch.rand(flat.shape, generator=gg) < fault['drop']) & sets_pre
    return (flat*(~drop).float()).view(B,T,U)
model.layers[1].ffn[1].register_forward_hook(fault_hook)

def loss_score(ids_batch, drop=0.0, seed_off=0):
    fault['drop']=drop; fault['seed']=seed_off
    with torch.no_grad():
        logits = model(ids_batch)
    fault['drop']=0.0
    lp = F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]), ids_batch[:,1:].reshape(-1), reduction='none').view(ids_batch.shape[0],-1)
    return lp.mean(dim=1).tolist()  # 每条序列的平均loss,越高越可疑

eval_g = torch.Generator().manual_seed(DATA_SEED + 424242)
normal_batches = [make_batch(ids_corpus, 16, SEQ_LEN, eval_g) for _ in range(N_SEQ//16)]
normal = []
for ids in normal_batches: normal += loss_score(ids)

out = {}
for dr in (0.1,0.3,0.5):
    s=[]
    for i,ids in enumerate(normal_batches): s += loss_score(ids, drop=dr, seed_off=i+1)
    a,se = auc(s, normal); out[f'A_drop{dr}']=(a,se); print(f'困惑度基线 A drop={dr}: AUC={a:.3f}±{se:.3f}', flush=True)
perm_g = torch.Generator().manual_seed(999)
s=[]
for ids in normal_batches:
    shuf = torch.stack([r[torch.randperm(SEQ_LEN, generator=perm_g)] for r in ids]); s += loss_score(shuf)
a,se = auc(s, normal); out['B']=(a,se); print(f'困惑度基线 B 乱序: AUC={a:.3f}±{se:.3f}', flush=True)
s=[]
for _ in range(N_SEQ//16):
    junk = torch.randint(1, V, (16, SEQ_LEN), generator=perm_g); s += loss_score(junk)
a,se = auc(s, normal); out['C']=(a,se); print(f'困惑度基线 C 乱码: AUC={a:.3f}±{se:.3f}', flush=True)
s=[]
for ids in normal_batches: s += loss_score(upper_map[ids])
a,se = auc(s, normal); out['D']=(a,se); print(f'困惑度基线 D 大写: AUC={a:.3f}±{se:.3f}', flush=True)
json.dump({k:list(v) for k,v in out.items()}, open(os.path.join(OUT_DIR,'expt10_baseline.json'),'w'))
print('完成')
