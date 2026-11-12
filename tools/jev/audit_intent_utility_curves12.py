import json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from queue_span_dp4_resume import check_pins
from prepare_intent_utility_curves import validate_protocol
from multimodal_jev.intent_actor_inputs import schedule
from multimodal_jev.slurp_intent_pilot import load_pilot
b=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev');root=b/'intent-utility-curves12-prepared-v1';read=lambda p:json.loads(p.read_text())
protocol=read(root/'protocol.json');pilot=read(Path(protocol['pilot_protocol']));validate_protocol(protocol,pilot)
check_pins(read(root/'prepared.json')['pins'])
items,_=load_pilot(protocol['pilot'],protocol['manifest_sha256'],'D')
assert protocol['schedule']==[[r['id'] for r in schedule(items,protocol['seed'],i,8)] for i in range(12)]
images=set();names=set();audits=[]
for arm in protocol['arms']:
 previous=None;previous_step=None;source=None
 for index,step in enumerate(arm['steps']):
  for kind,devices in [('training_root',[4,5,6,7]),('suite_root',[4])]:
   p=Path(step[kind]);plan=read(p/'prepared.json');cfg=read(p/'config-template.json');images.add(plan['image'])
   assert plan['physical_devices']==devices and plan['root']==str(p)
   assert plan['config_template_sha256']==file_hash(p/'config-template.json')
   assert plan['source_manifest_sha256']==file_hash(p/'source-manifest.json')
   assert not any((p/n).exists() for n in ['config.json','launch.json','create-attempt.json','run'])
   name=plan['create_argv'][plan['create_argv'].index('--name')+1];assert name not in names;names.add(name)
   manifest=read(p/'source-manifest.json')
   if source is None:source=manifest
   assert manifest==source
   check_pins(cfg['pins']);check_pins({str(p/'source'/k):h for k,h in manifest.items()})
  cfg=read(Path(step['training_root'])/'config-template.json')
  assert cfg['step']==index and cfg['objective']==arm['objective']
  if previous is not None:
   validate_lineage(previous,cfg)
   assert cfg['previous_root']==previous_step['training_root'] and cfg['previous_suite']==previous_step['suite_root']
  suite=read(Path(step['suite_root'])/'config-template.json')
  assert suite['training_root']==step['training_root'] and suite['adapter']==str(Path(step['training_root'])/'run/adapter')
  previous=cfg;previous_step=step
 audits.append(dict(arm=arm['name'],updates=12,source_uses=96,source_manifest_preserved=True))
assert images=={'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'}
result=dict(passed=True,observed_unix=time.time(),root=str(root),arms=audits,distinct_container_names=len(names),image=next(iter(images)),npu_dispatched=False,
 files={str(root/n):file_hash(root/n) for n in ['protocol.json','prepared.json']},scope='Actual prepared configs, source snapshots, all known pins and schedule independently checked. Future results are not invented; no training or independent efficacy claim.')
atomic_json(root/'independent-preparation-audit.json',result);atomic_json(Path('docs/engineering/intent-utility-curves12-prepared-v1-verified.json'),result)
print(result)
