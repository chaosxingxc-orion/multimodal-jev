#!/usr/bin/env python3
"""Offline padding simulation from authenticated full processor records, no training."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('.local/simple-final-v1'));parser.add_argument('--output',type=Path,default=Path('docs/engineering/slurp-length-bucketing-analysis-v1.json'));args=parser.parse_args()
root=args.root; raw=(root/'processor-v2-records.jsonl').read_bytes(); proof=json.loads((root/'processor-v2-verified.json').read_text())
assert hashlib.sha256(raw).hexdigest()==proof['records_sha256']
records=[json.loads(s) for s in raw.splitlines()]; assert len(records)==81026
byid={r['id']:r for r in records if r['epoch']==0};assert len(byid)==40513
for r in records:assert (r['input_tokens'],r['samples'])==(byid[r['id']]['input_tokens'],byid[r['id']]['samples'])
manifest_raw=(root/'schedule/MANIFEST.json').read_bytes();manifest=json.loads(manifest_raw)
processor=json.loads((root/'processor-v2-result.json').read_text())
assert hashlib.sha256(manifest_raw).hexdigest()==processor['schedule_sha256']=='fc36b3afdf367bad6a5f0530a8743cb44b630592e2c778c3046130e3c1adf200'
for e in manifest['epochs']:
 assert hashlib.sha256((root/f"schedule/epoch-{e['epoch']}.json").read_bytes()).hexdigest()==e['file_sha256']
schedules=[json.loads((root/f'schedule/epoch-{e}.json').read_text())['batches'] for e in range(2)]
coords={(r['epoch'],r['batch'],r['slot']):r for r in records}
for e,batches in enumerate(schedules):
 for b,slots in enumerate(batches):
  assert len(slots)==128
  for i,s in enumerate(slots):
   if not s['padding']:assert coords[e,b,i]['id']==s['id']
def summary(vals):
 vals=sorted(vals);return {str(q):vals[round((len(vals)-1)*q)] for q in [0,.1,.25,.5,.75,.9,.95,.99,1]}
def evaluate(micro,order):
 token_used=token_pad=audio_used=audio_pad=0;ratios=[]
 for batches in schedules:
  for slots in batches:
   ids=[s['id'] for s in slots]
   if order=='rank_original':ranks=[ids[r*32:(r+1)*32] for r in range(4)]
   elif order=='rank_sorted':ranks=[sorted(ids[r*32:(r+1)*32],key=lambda i:(byid[i]['input_tokens'],byid[i]['samples'],i)) for r in range(4)]
   else:
    ids=sorted(ids,key=lambda i:(byid[i]['input_tokens'],byid[i]['samples'],i));ranks=[[] for _ in range(4)]
    for start in range(0,128,4*micro):
     for r in range(4):
      dst=r if (start//(4*micro))%2==0 else 3-r
      ranks[dst]+=ids[start+r*micro:start+(r+1)*micro]
   assert sorted(i for rank in ranks for i in rank)==sorted(s['id'] for s in slots)
   rankcost=[]
   for rank in ranks:
    cost=0
    for st in range(0,32,micro):
     group=[byid[i] for i in rank[st:st+micro]];ls=[x['input_tokens'] for x in group];au=[x['samples'] for x in group]
     token_used+=sum(ls);token_pad+=micro*max(ls);audio_used+=sum(au);audio_pad+=micro*max(au);cost+=micro*max(ls)**2
    rankcost.append(cost)
   ratios.append(max(rankcost)/statistics.mean(rankcost))
 return dict(micro=micro,order=order,token_padding_fraction=1-token_used/token_pad,waveform_padding_proxy_fraction=1-audio_used/audio_pad,padded_token_slots=token_pad,mean_slowest_rank_over_mean_dense_attention_proxy=statistics.mean(ratios))
result=dict(schedule_sha256=processor['schedule_sha256'],population=40513,epochs=2,records_sha256=proof['records_sha256'],token_quantiles=summary([r['input_tokens'] for r in byid.values()]),duration_seconds_quantiles=summary([r['samples']/16000 for r in byid.values()]),token_bucket_counts={str(hi):sum(lo<r['input_tokens']<=hi for r in byid.values()) for lo,hi in [(0,1984),(1984,2048),(2048,2112),(2112,2249)]},comparisons=[evaluate(m,o) for m in [8,16,32] for o in ['rank_original','rank_sorted','global_sorted_snake']],scope='Offline full two-epoch schedule simulation, same128 slot membership and weights per update including zero-weight padding; waveform padding is a proxy, not audio encoder FLOPs. No NPU execution or speedup claim. Token records authenticated; schedule coordinate and multiplicity checked.')
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
