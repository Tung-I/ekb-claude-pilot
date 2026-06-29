import json, numpy as np
from collections import OrderedDict, Counter
R=[json.loads(l) for l in open('scratch_adn/all_records.jsonl')]
kb=[r for r in R if r['run']=='popqa_kb' and r['s_pop'] and r['cost'] is not None]
rng=np.random.default_rng(0)

key=[ (r['subj'],r['prop']) for r in kb ]
pop=np.array([r['s_pop'] for r in kb],float)
cost=np.array([r['cost'] for r in kb],float)
correct=np.array([1 if r['correct'] else 0 for r in kb])
n=len(kb)

def make_stream(M, alpha):
    w=pop**alpha; w/=w.sum()
    return rng.choice(n, size=M, p=w)

M=200000
for alpha in [1.0, 0.7]:
    stream=make_stream(M,alpha)
    # infinite cache
    seen=set(); hits=0; saved=0.0; served_correct=0; false_hits=0; served=0
    # finite caches LRU & LFU
    for C in [None]:
        pass
    for i in stream:
        k=key[i]
        if k in seen:
            hits+=1; saved+=cost[i]; served+=1
            if correct[i]==1: served_correct+=1
            else: false_hits+=1
        else:
            seen.add(k)
    print(f"alpha={alpha}  M={M:,}  unique={len(seen):,}/{n}")
    print(f"  INF cache: hitrate={hits/M:.1%}  cost_saved={saved/ (cost.mean()*M):.1%} of total  "
          f"served_acc_on_hits={served_correct/max(1,served):.1%}  false_hit_rate(of all req)={false_hits/M:.1%}")

# finite cache sweep (LFU and LRU) at alpha=1.0
stream=make_stream(M,1.0)
def sim_finite(stream, C, policy):
    if policy=='LFU':
        cache={}  # k -> freq
        order=None
    cacheset=OrderedDict()  # k -> count (for LFU) ; for LRU use move_to_end
    freq=Counter()
    hits=0
    for i in stream:
        k=key[i]
        if k in cacheset:
            hits+=1
            if policy=='LRU': cacheset.move_to_end(k)
            if policy=='LFU': freq[k]+=1
        else:
            if len(cacheset)>=C:
                if policy=='LRU':
                    cacheset.popitem(last=False)
                else: # LFU evict min freq
                    ek=min(cacheset, key=lambda x:freq[x]); del cacheset[ek]
            cacheset[k]=1; freq[k]+=1
    return hits/len(stream)
print("\nFinite-cache hit rate (alpha=1.0, iid popularity, no temporal locality):")
print(f"{'capacity':>9}{'LRU':>8}{'LFU':>8}  (catalog=9756)")
for C in [100,500,1000,2000,5000]:
    print(f"{C:>9}{sim_finite(stream,C,'LRU'):>8.1%}{sim_finite(stream,C,'LFU'):>8.1%}")
