"""Bind the existing bounded DP4 trainer to the frozen attribute experiment.

No NPU launch here. Source derivation is exact and fails on upstream changes.
Execution remains limited to two updates, with the original collective restore,
FP32 optimizer, BF16 export, rollout activation and numerical parity checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import types

from multimodal_jev.baseline import file_hash
from tools.reports.context_attribute_actor_candidate import generate as actor_source, digest
from tools.reports.context_attribute_training_contract import load_attribute_schedule, checkpoint_contract

TRAINER_SHA='7bb1b2259bd799907856455cf75475280dd0bbb3998f6a928c03595141e73288'
DATA_SHA='d98db48e619a83e7eceb308da13f257a6cc3cfa4bf701f3f17022ff99a07a482'
BRIDGES={
    'tools/reports/context_attribute_training_contract.py':'04dd13d6d5663333f82da8f2b14ee5f98ecdac4f2e12e77bc1495c8134645ff6',
    'tools/reports/context_attribute_schedule_candidate.py':'2dfdb901b73ac30e79e7c0e0208ece5b57530aedc3467065e2eaac5ad2e11c69',
    'tools/reports/context_attribute_input_candidate.py':'9e16e7f6f653b0a1d9f2c218609e722617eab69c5d965f29b624bc253eee0e94',
}


def load_dataset(root):
    root=Path(root)
    if file_hash(root/'MANIFEST.json')!=DATA_SHA:
        raise ValueError('Exact full TRAIN attribute candidate required')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,h in manifest['files'].items():
        if file_hash(root/name)!=h:raise ValueError('Attribute dataset bytes changed: '+name)
    menu=json.loads((root/'catalog.json').read_text())
    inputs=[json.loads(s) for s in (root/'inputs.jsonl').read_text().splitlines()]
    targets=[json.loads(s) for s in (root/'targets.jsonl').read_text().splitlines()]
    if (len(inputs)!=40513 or len(targets)!=40513 or len({r['id'] for r in inputs})!=40513
            or len({r['id'] for r in targets})!=40513 or digest(menu)!=manifest['catalog_sha256']):
        raise ValueError('Complete frozen attribute population required')
    return menu,inputs,targets,manifest


def validate_targets(menu, inputs, targets, population):
    """Reconstruct every attribute from clean source labels, before prefetch."""
    nodes=menu['nodes'];ids=[n['id'] for n in nodes];catalog_id=digest(menu)
    projections={e['id']:[e['id'] in n['members'] for n in nodes] for e in menu['events']}
    byid={r['id']:r for r in targets}
    if (len(byid)!=len(targets) or set(byid)!={r['id'] for r in inputs}
            or set(byid)!=set(population)):
        raise ValueError('Complete source/input/target identity required')
    for asset in inputs:
        uid=asset['id'];row=population[uid];target=byid[uid]
        if (set(asset)!={'id','audio','audio_sha256','catalog_sha256'}
                or asset['audio']!=row['audio'] or asset['audio_sha256']!=row['audio_sha256']
                or asset['catalog_sha256']!=catalog_id or row['intent'] not in projections
                or row['role']!='learning' or row['official_split']!='train' or row['known_pending_source'] is not False
                or target['input_sha256']!=digest(asset) or target['source_group']!=row['source_group']
                or target['candidate_ids']!=ids
                or len(target['labels'])!=len(nodes) or len(target['supervised'])!=len(nodes)
                or any(type(x) is not bool for x in target['labels']+target['supervised'])
                or not all(target['supervised'])
                or target['labels']!=projections[row['intent']]
                or target['truth_basis']!=menu['supervision']):
            raise ValueError('Native full TRAIN projection or audio binding differs: '+uid)


class AttributeTrainingRuntime:
    """Authenticated task inputs; original trainer owns device and lifecycle."""
    def __init__(self, config):
        self.config=copy.deepcopy(config);c=self.config;self.source_root=Path(c['attribute_source_root'])
        if (c.get('engineering_prefix') is not True or c.get('training_dispatch_allowed') is not False
                or any(type(c[k]) is not int for k in ('start_update','stop_update'))
                or not 0<=c['start_update']<c['stop_update']<=634
                or c['stop_update']-c['start_update']>2
                or c['physical_devices']!=[4,5,6,7] or c['checkpoint_interval']!=1
                or c.get('binary_policy_arithmetic')!='context-bernoulli-probability-cpu-log-v2'
                or c['behavior_bounds']!={'probability_max_abs':2e-7,'binary_logprob_max_abs':2e-6}):
            raise ValueError('Bounded exact-parity attribute engineering configuration required')
        for path,h in c['pins'].items():
            if file_hash(path)!=h:raise ValueError('Candidate dependency changed: '+path)
        for name,h in BRIDGES.items():
            if file_hash(self.source_root/name)!=h:raise ValueError('Attribute bridge changed: '+name)
        self.menu,self.inputs,self.targets,self.manifest=load_dataset(c['attribute_dataset'])
        self.arm=c['attribute_arm']
        _,self.actor_contract=actor_source(self.menu,self.arm,source_root=self.source_root)
        from multimodal_jev.slurp_official_training import load
        rows,_,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
        self.population={r['id']:r for r in rows}
        if len(rows)!=40513 or len(self.population)!=40513:raise ValueError('Full TRAIN source required')
        validate_targets(self.menu,self.inputs,self.targets,self.population)
        self.batches,self.schedule=load_attribute_schedule(c['attribute_schedule'],c['attribute_schedule_sha256'],
            original_schedule=c['schedule'],population=self.population,menu=self.menu,
            arm=self.arm,source_root=self.source_root)
        if (self.schedule['source_schedule_sha256']!=c['schedule_sha256']
                or self.schedule['candidate_manifest_sha256']!=DATA_SHA
                or self.manifest['release_sha256']!=c['release_sha256']):
            raise ValueError('Source schedule or TRAIN release differs')
        self.contract=checkpoint_contract(c,self.menu,self.arm,self.schedule,source_root=self.source_root)
        self.contract['dataset_manifest_sha256']=DATA_SHA
        self.contract['trainer_source_sha256']=TRAINER_SHA
        self.contract['derived_trainer_sha256']=derive(self.source_root)[1]['derived_sha256']
        self.contract['attribute_bridges']=dict(BRIDGES)
        self.contract['attribute_runtime_sha256']=file_hash(__file__)

    def actor(self):
        from tools.reports.context_attribute_actor_candidate import load
        return load(self.menu,self.arm,source_root=self.source_root)[0]

    def full_schedule(self,population):
        if population!=self.population:raise ValueError('Trainer source population differs')
        return copy.deepcopy(self.batches),copy.deepcopy(self.schedule)

    def stream(self,factory,cache,**kwargs):
        from tools.reports.context_attribute_input_candidate import AttributeInputs
        from tools.reports.context_attribute_schedule_candidate import AttributeScheduledBatches
        preparer=AttributeInputs(self.menu,self.arm,source_root=self.source_root,
            processor_factory=factory,cache=cache,**kwargs)
        stream=AttributeScheduledBatches(self.batches,self.inputs,self.targets,preparer,
            source_root=self.source_root,start_update=self.config['start_update'],
            stop_update=self.config['stop_update'],global_batch=128)
        return preparer,stream

    def session(self,client,cache,preparer,**kwargs):
        from tools.reports.context_attribute_schedule_candidate import AttributeRolloutSession
        return AttributeRolloutSession(client,cache,input_adapter=preparer,source_root=self.source_root,**kwargs)


def derive(source_root):
    path=Path(source_root)/'tools/jev/train_context_online_candidate.py'
    if file_hash(path)!=TRAINER_SHA:raise ValueError('Pinned bounded trainer changed')
    original=path.read_text();text=original
    old_contract="""        contract={k:c[k] for k in ('base_id','source_policy_id','source_contract','runtime_profile',
            'release_sha256','schedule_sha256','global_batch','microbatch','stage','algorithm','behavior_bounds')}
        contract.update(optimizer=dict(lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.),
            objective=dict(group_size=16,reward='complete_candidate_set_f1',kl=.01,auxiliary='cosine_0.3_to_0.1_first_80pct'),grad_clip=1.,world_size=4)"""
    old_stream="""                preparer=CachedContextPreparer(factory,EncodedAudioCache(c['encoded_audio_cache']),
                    processor_sha256=processor_identity,decision_token_id=model.decision_token_id,
                    audio_token_id=base.config.audio_token_id,max_tokens=4096,audio_mirror=audio_mirror)
                stream=CachedContextBatches(batches,population,task,preparer,
                    start_update=start_update,stop_update=stop_update,global_batch=128)"""
    replacements=[
        ('    from multimodal_jev.context_native_actor import MODE,build_actor,make_batch',
         '    attribute_actor=ATTRIBUTE_RUNTIME.actor()\n    MODE,build_actor,make_batch=attribute_actor.MODE,attribute_actor.build_actor,attribute_actor.make_batch'),
        (old_contract,'        contract=copy.deepcopy(ATTRIBUTE_RUNTIME.contract)'),
        ("                batches,manifest=load_schedule(c['schedule'],c['schedule_sha256'],\n                    release_sha256=c['release_sha256'],population=population)",
         '                batches,manifest=ATTRIBUTE_RUNTIME.full_schedule(population)'),
        (old_stream,"""                preparer,stream=ATTRIBUTE_RUNTIME.stream(factory,EncodedAudioCache(c['encoded_audio_cache']),
                    processor_sha256=processor_identity,decision_token_id=model.decision_token_id,
                    audio_token_id=base.config.audio_token_id,max_tokens=4096,audio_mirror=audio_mirror)"""),
        ('                session=ContextRolloutSession(client,cache,reference=reference,initial_policy=initial,',
         '                session=ATTRIBUTE_RUNTIME.session(client,cache,preparer,reference=reference,initial_policy=initial,'),
        ("        cfg.jev_context_objective=options", "        options['attribute_contract_id']=ATTRIBUTE_RUNTIME.actor_contract['contract_id']\n        cfg.jev_context_objective=options"),
        ("groups=[{k:p[k] for k in ('request_id','policy_id','labels'", "groups=[{k:p[k] for k in ('attribute_contract_id','decision_ids','request_id','policy_id','labels'"),
    ]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('Trainer adaptation anchor changed: '+old)
        text=text.replace(old,new)
    compile(text,'<attribute-bounded-trainer>','exec')
    return text,dict(original_sha256=TRAINER_SHA,derived_sha256=hashlib.sha256(text.encode()).hexdigest(),
        substitutions=len(replacements),training_dispatch_allowed=False)


def run(config,output):
    # No remote launch; caller must supply an independently leased fixed-image
    # DP4 container and service configured for 124 readouts (60 for Direct).
    runtime=AttributeTrainingRuntime(config)
    source,receipt=derive(runtime.source_root)
    module=types.ModuleType('attribute_bounded_trainer')
    module.ATTRIBUTE_RUNTIME=runtime
    exec(compile(source,'<attribute-bounded-trainer>','exec'),module.__dict__)
    return module.run(runtime.config,Path(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
