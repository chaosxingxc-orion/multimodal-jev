import json,hashlib,subprocess,shlex
from pathlib import Path
from collections import defaultdict,Counter
from bisect import bisect_right
R=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev/audio2tool-jev-context-v1')
E=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
launch=read(R/'launch.json');intent=read(R/'intent.json');result=read(R/'run/result.json')
assert sha(R/'intent.json')==launch['intent_sha256']=='c6d137d805c7d4d3ec8bf7331dcb90f32970e9112ca507898e608d2a5c80cc15'
x=json.loads(subprocess.check_output(['ssh','npu',shlex.join(['docker','inspect',launch['container_id']])]))[0]
assert x['Id']==launch['container_id'] and x['Image']==launch['image']==intent['image']
assert x['State']['Status']=='exited' and x['State']['ExitCode']==0 and not x['State']['OOMKilled']
assert read(R/'run/terminal.json')=={'exit_code':0}
h=x['HostConfig'];assert h['NanoCpus']==2_000_000_000 and h['Memory']==20*1024**3 and h['NetworkMode']=='none' and not h['Devices']
argv=intent['argv'];assert x['Config']['Cmd']==argv[argv.index(intent['image'])+1:]
expected_binds=[argv[i+1] for i,a in enumerate(argv) if a=='-v'];assert h['Binds']==expected_binds
expected_env=[argv[i+1] for i,a in enumerate(argv) if a=='-e'];assert all(e in x['Config']['Env'] for e in expected_env)
for n,pin in intent['source_files'].items():
 assert sha(R/'source'/n)==pin
 assert hashlib.sha256(subprocess.check_output(['git','show',intent['source_commit']+':'+n])).hexdigest()==pin
for n,pin in result['source_sha256'].items():assert intent['source_files'][n]==pin
assert sha(R/'run/profiles.jsonl')==result['profiles_sha256']
release=Path(argv[argv.index('--release')+1]);manifest=read(release/'MANIFEST.json')
assert sha(release/'MANIFEST.json')==result['manifest_sha256']=='56921792b06636d83bd6b8a21e49ae0f887f3de68a9813d75fe1b98d83c0f062'
for name,key in [('task-bank.json','task_bank_file_sha256'),('external_test.requests.jsonl','requests_sha256')]:assert sha(release/name)==manifest[key]
assert sha(E/'audio2tool-native-inputs-v1/inputs.jsonl')=='f7e4ebe4e7cf2cf411f3f26a8714bbdd0a493c0c7aeb98dd4199112d197e1a35'
assert sha(E/'audio2tool-native-context-v1/run/budgets.jsonl')=='87e750243a3dcc56ca006e9f661dcaa2ac130cc27fd7312d6e986892aa1ef223'
rows=[json.loads(l) for l in (E/'audio2tool-native-inputs-v1/inputs.jsonl').open()]
budgets=[json.loads(l) for l in (E/'audio2tool-native-context-v1/run/budgets.jsonl').open()]
profiles=[json.loads(l) for l in (R/'run/profiles.jsonl').open()]
bank=read(release/'task-bank.json');decision={d['id']:d for d in bank['decisions']}
groups=defaultdict(list)
for row,budget in zip(rows,budgets,strict=True):
 assert row['id']==budget['id']
 inp=row['input'];groups[(inp['mode'],tuple((a.get('turn'),a.get('role')) for a in inp['audios']))].append(budget)
assert len(rows)==30733 and len(groups)==6 and len(decision)==457 and len(profiles)==2742
seen=set();computed={str(n):Counter() for n in [4096,8192,16384,32768]};maxima={};maximum=0
for p in profiles:
 key=(p['mode'],tuple(tuple(r) for r in p['roles']));members=groups[key];pair=(key,p['decision_id'])
 assert pair not in seen;seen.add(pair)
 assert p['records']==len(members) and p['decision_type']==decision[p['decision_id']]['decision_type']
 lengths=sorted(sum(b['audio_tokens']) for b in members);n=p['text_tokens']+lengths[-1];assert n==p['max_input_tokens'];maximum=max(maximum,n)
 for limit,c in computed.items():c[p['decision_type']]+=len(lengths)-bisect_right(lengths,int(limit)-1024-p['text_tokens'])
 for category in [p['mode']]+(['seven_audio'] if len(key[1])==7 else []):
  k=category+':'+p['decision_type'];maxima[k]=max(maxima.get(k,0),n)
assert {k:dict(v) for k,v in computed.items()}==result['exceeded_with_1024_output']
assert maximum==result['max_input_tokens']==18168 and result['decisions']==30733*457
assert result['requests']==30733 and result['profiles']==2742 and result['layouts']==6 and result['status']=='passed'
assert result['model_loaded'] is False and result['targets_read'] is False
cases=result['real_processor_cases'];assert len(cases)==12 and {c['case'] for c in cases}==set(maxima)
budget_by_id={b['id']:b for b in budgets};row_by_id={r['id']:r for r in rows}
for case in cases:
 assert read(R/'run'/(case['case'].replace(':','-')+'.json'))==case
 assert case['input_tokens']==maxima[case['case']] and case['token_ids_exact'] and case['bf16_features_exact']
 assert case['frames']==budget_by_id[case['id']]['frames']
 assert len(case['audio_receipts'])==len(row_by_id[case['id']]['input']['audios'])
 for receipt,audio in zip(case['audio_receipts'],row_by_id[case['id']]['input']['audios'],strict=True):
  assert receipt['path']==audio['path'] and receipt['sha256']==audio['sha256'] and receipt['bytes']==audio['bytes']
compact={k:v for k,v in result.items() if k!='real_processor_cases'}
compact['real_processor_cases']=[{k:v for k,v in c.items() if k!='audio_receipts'} for c in cases]
report=dict(root=str(R),launch=launch,state=x['State'],result=compact,result_sha256=sha(R/'run/result.json'),
 verifier_sha256=sha(Path(__file__)),source_files_verified=len(intent['source_files']),
 verification='Live exact-container inspection; pinned launch intent, Cmd, binds, env, CPU/memory/network/devices; snapshot hashes against Git commit; full 2742-profile independent budget arithmetic over 30733 native rows; all 12 individual case files and source audio receipt bindings. Does not rerun processors or claim NPU engine capacity.')
p=Path('docs/engineering/audio2tool-jev-context-v1-verified.json')
with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'result_sha256':report['result_sha256'],'maximum':maximum,'overflows':report['result']['exceeded_with_1024_output'],'cases':len(cases),'elapsed_seconds':result['elapsed_seconds']}))
