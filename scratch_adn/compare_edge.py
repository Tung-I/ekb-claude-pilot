import json, glob, statistics as st
# absolute $/M-token prices
PRICE={'sonnet':dict(i=3,o=15,cw=3.75,cr=0.30), 'haiku':dict(i=1,o=5,cw=1.25,cr=0.10)}
def dollars(u,model):
    if not isinstance(u,dict): return None
    p=PRICE[model]
    return ((u.get('input_tokens',0) or 0)*p['i']+(u.get('output_tokens',0) or 0)*p['o']
           +(u.get('cache_creation_input_tokens',0) or 0)*p['cw']
           +(u.get('cache_read_input_tokens',0) or 0)*p['cr'])/1e6
def load(run,model):
    out={}
    for f in glob.glob(f'traces/claude_native/{run}/*/normalized_trace.json'):
        d=json.load(open(f)); out[d['query_id']]={'correct':bool(d.get('any_match')),
          'cost':dollars(d.get('usage') or {},model),'conf':d.get('confidence'),
          'n_ws':sum(1 for s in d.get('steps',[]) if s.get('tool')=='WebSearch')}
    return out
edge=load('popqa_edge_haiku_pilot','haiku'); orig=load('popqa_test','sonnet')
ids=[i for i in edge if i in orig]
ea=[edge[i]['correct'] for i in ids]; oa=[orig[i]['correct'] for i in ids]
ec=[edge[i]['cost'] for i in ids]; oc=[orig[i]['cost'] for i in ids]
print(f"matched={len(ids)}  (REAL $ per task)")
print(f"EDGE  (haiku) : acc={st.mean(ea):.0%}  mean=${st.mean(ec)*1000:.3f}/1k-tasks*  mean_cost=${st.mean(ec):.5f}  med_ws={st.median([edge[i]['n_ws'] for i in ids])}")
print(f"ORIGIN(sonnet): acc={st.mean(oa):.0%}  mean_cost=${st.mean(oc):.5f}  med_ws={st.median([orig[i]['n_ws'] for i in ids])}")
print(f"edge/origin cost ratio = {st.mean(ec)/st.mean(oc):.2f}  (edge is {1-st.mean(ec)/st.mean(oc):.0%} cheaper)")
both=sum(1 for i in ids if edge[i]['correct'] and orig[i]['correct'])
eo=sum(1 for i in ids if edge[i]['correct'] and not orig[i]['correct'])
oe=sum(1 for i in ids if not edge[i]['correct'] and orig[i]['correct'])
nn=sum(1 for i in ids if not edge[i]['correct'] and not orig[i]['correct'])
print(f"\nconfusion: both={both} edge-only={eo} origin-only={oe} both-wrong={nn}")
# policies
def policy_cost_acc(route_edge):  # route_edge: list bool
    c=sum(ec[k] if route_edge[k] else oc[k] for k in range(len(ids)))
    a=sum(1 for k in range(len(ids)) if (ea[k] if route_edge[k] else oa[k]))/len(ids)
    return c,a
c,a=policy_cost_acc([True]*len(ids));  print(f"\nALWAYS-EDGE  : acc={a:.0%} cost=${c:.4f} ({c/sum(oc):.0%} of origin)")
c,a=policy_cost_acc([False]*len(ids)); print(f"ALWAYS-ORIGIN: acc={a:.0%} cost=${c:.4f} (100%)")
# oracle: edge if edge correct else origin
orc=[ea[k] for k in range(len(ids))]; c,a=policy_cost_acc(orc)
print(f"ORACLE route : acc={a:.0%} cost=${c:.4f} ({c/sum(oc):.0%} of origin)")
# confidence-gated: route edge if haiku conf high
hi=[edge[i]['conf']=='high' for i in ids]; c,a=policy_cost_acc(hi)
print(f"CONF-gated   : acc={a:.0%} cost=${c:.4f} ({c/sum(oc):.0%} of origin)  [edge if haiku conf=high; {sum(hi)}/{len(ids)} to edge]")
