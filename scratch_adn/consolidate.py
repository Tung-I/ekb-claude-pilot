import json, glob, os
from pathlib import Path

# Sonnet pricing ratios in input-token-equivalent units
def cost_tokens(u):
    if not isinstance(u,dict): return None
    it=u.get('input_tokens',0) or 0
    ot=u.get('output_tokens',0) or 0
    cc=u.get('cache_creation_input_tokens',0) or 0
    cr=u.get('cache_read_input_tokens',0) or 0
    return it*1 + ot*5 + cc*1.25 + cr*0.1

# popqa metadata
meta={}
for fn in ['data/popqa/popqa_filtered_kb.jsonl','data/popqa/popqa_filtered_test.jsonl']:
    if os.path.exists(fn):
        for line in open(fn):
            line=line.strip()
            if line:
                d=json.loads(line); meta[d['task_id']]=d

def ws_query(steps):
    for s in steps:
        if s.get('tool')=='WebSearch':
            ad=s.get('action_detail') or {}
            if isinstance(ad,dict): return ad.get('query')
    return None

runs={
 'popqa_kb':'popqa','popqa_test':'popqa',
 'frames_kb':'frames','frames_test':'frames',
 'gaia_lv1_x3':'gaia1','gaia_lv2_x3':'gaia2',
}
out=open('scratch_adn/all_records.jsonl','w')
N=0
for run,bench in runs.items():
    for f in glob.glob(f'traces/claude_native/{run}/*/normalized_trace.json'):
        d=json.load(open(f))
        steps=d.get('steps',[])
        toolseq=[s.get('tool') for s in steps]
        subst=[t for t in toolseq if t not in ('ToolSearch','StructuredOutput')]
        m=meta.get(d['query_id'],{})
        u=d.get('usage') or {}
        rec={
          'tid':d['query_id'],'run':run,'bench':bench,'q':d.get('query_text'),
          'prop':d.get('prop'),'subj':m.get('subj'),'obj':m.get('obj'),
          's_pop':m.get('s_pop'),'o_pop':m.get('o_pop'),
          'tokens_raw':d.get('total_tokens'),'cost':cost_tokens(u),
          'cache_read':u.get('cache_read_input_tokens'),'output':u.get('output_tokens'),
          'n_tool':d.get('total_tool_calls'),'n_ws':sum(1 for t in toolseq if t=='WebSearch'),
          'n_wf':sum(1 for t in toolseq if t=='WebFetch'),
          'toolseq':toolseq,'subst':subst,'subst_str':'>'.join(subst),
          'ws_query':ws_query(steps),
          'pred':d.get('final_answer_pred'),
          'em':d.get('exact_match'),'any_match':d.get('any_match'),
        }
        rec['correct']= rec['any_match'] if bench=='popqa' else rec['em']
        out.write(json.dumps(rec,ensure_ascii=False)+'\n'); N+=1
out.close()
print("wrote",N,"records to scratch_adn/all_records.jsonl")
