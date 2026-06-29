import json, glob, statistics as st
PRICE={'sonnet':dict(i=3,o=15,cw=3.75,cr=0.30),'haiku':dict(i=1,o=5,cw=1.25,cr=0.10)}
def dollars(u,m):
    p=PRICE[m]; 
    return ((u.get('input_tokens',0)or 0)*p['i']+(u.get('output_tokens',0)or 0)*p['o']+(u.get('cache_creation_input_tokens',0)or 0)*p['cw']+(u.get('cache_read_input_tokens',0)or 0)*p['cr'])/1e6
def load(run,m):
    out={}
    for f in glob.glob(f'traces/claude_native/{run}/*/normalized_trace.json'):
        d=json.load(open(f)); out[d['query_id']]={'correct':bool(d.get('exact_match')),'cost':dollars(d.get('usage')or{},m),
          'nws':sum(1 for s in d.get('steps',[]) if s.get('tool')=='WebSearch'),'nwf':sum(1 for s in d.get('steps',[]) if s.get('tool')=='WebFetch')}
    return out
edge=load('frames_edge_haiku_pilot','haiku'); orig=load('frames_test','sonnet')
ids=[i for i in edge if i in orig]
ea=[edge[i]['correct'] for i in ids]; oa=[orig[i]['correct'] for i in ids]
ec=[edge[i]['cost'] for i in ids]; oc=[orig[i]['cost'] for i in ids]
print(f"FRAMES matched={len(ids)}  (strict EM, real $)")
print(f"EDGE  (haiku) : EM={st.mean(ea):.0%} mean$={st.mean(ec):.5f} med_ws={st.median([edge[i]['nws'] for i in ids])} med_wf={st.median([edge[i]['nwf'] for i in ids])}")
print(f"ORIGIN(sonnet): EM={st.mean(oa):.0%} mean$={st.mean(oc):.5f} med_ws={st.median([orig[i]['nws'] for i in ids])} med_wf={st.median([orig[i]['nwf'] for i in ids])}")
print(f"cost ratio edge/origin={st.mean(ec)/st.mean(oc):.2f}")
print(f"confusion both={sum(1 for k in range(len(ids)) if ea[k] and oa[k])} edge-only={sum(1 for k in range(len(ids)) if ea[k] and not oa[k])} origin-only={sum(1 for k in range(len(ids)) if oa[k] and not ea[k])} both-wrong={sum(1 for k in range(len(ids)) if not ea[k] and not oa[k])}")
