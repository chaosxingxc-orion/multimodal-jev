"""Initialize, certify token layouts, and run admitted Stage1 windows on the selected host."""
import json
import math
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.context_stage_lora import bind_design


def read(path):
    return json.loads(Path(path).read_bytes())


def prepare(c, root):
    from tools.reports.initialize_context_stage1_policy import run
    from multimodal_jev.stage1_card_layout import from_panel
    from multimodal_jev.context_decision_contract import encode_context_request
    from multimodal_jev.stage1_layout_cache import split_text
    from multimodal_jev.audio_policy import load_audio
    from transformers import Qwen3ASRProcessor
    entry = Path(c['entry'])
    design = entry/'design.json'
    init = dict(read(entry/'init.json'),design=str(design),design_sha256=file_hash(design))
    policy = run(init,root)
    native = read(entry/'native-cache.json')
    plan = read(entry/'card-plan.json')
    processor = Qwen3ASRProcessor.from_pretrained(init['processor'],local_files_only=True)
    encoding = {k:v for k,v in native['encoding'].items() if k in ('audio_token_id','decision_token_id','prompt_layout','max_tokens')}
    sample = c['sample']
    if file_hash(sample['path']) != sample['sha256']:
        raise ValueError('processor witness audio changed')
    audio = load_audio(sample['path'],None)
    templates = {}
    for arm,panel in plan['panels'].items():
        batch,positions = encode_context_request(processor,from_panel(panel),audio,**encoding)
        prefix,suffix,_ = split_text(batch['input_ids'][0],encoding['audio_token_id'])
        templates[arm] = dict(panel=panel,prefix=prefix,suffix=suffix)
    if templates['native'] != native['templates']['native']:
        raise ValueError('processor reproduction of native cache differs')
    lengths = read(entry/'native-lengths.json')
    if lengths.keys() != plan['assignments'].keys():
        raise ValueError('missing full-population native lengths')
    deltas = {k:len(t['prefix'])+len(t['suffix'])-len(templates['native']['prefix'])-len(templates['native']['suffix'])
              for k,t in templates.items()}
    maximum = max(lengths[uid]+deltas[arm] for uid,pair in plan['assignments'].items() for arm in pair)
    if maximum > encoding['max_tokens']:
        raise ValueError(f'full input exceeds configured length: {maximum}')
    cache = dict(native,schema='jev-card-text-cache-v1', templates=templates,layout_plan_sha256=file_hash(entry/'card-plan.json'),
                 full_verified=len(lengths),panel_verified=len(templates),max_tokens=maximum)
    commit_json(root/'card-cache.json',cache)
    commit_json(root/'receipt.json',dict(policy_id=policy['policy_id'],cache_sha256=file_hash(root/'card-cache.json'),
                                        population=len(lengths),max_tokens=maximum,panels=len(templates),npu_executed=False))


def run(c):
    role = c['role']
    entry, work = Path(c['entry']),Path(c['work'])
    root = work/role
    root.mkdir()
    reuse = c.get('prepared_input')
    init = Path(reuse['root']) if reuse else work/'prepare'
    acoustic = Path(c['prepared_mixed']['root']) if 'prepared_mixed' in c else work/'prepare'
    if role == 'prepare':
        if 'prepared_mixed' in c:
            receipt = acoustic/'receipt.json'
            if file_hash(receipt)!=c['prepared_mixed']['receipt_sha256']:
                raise ValueError('mixed preparation receipt changed')
            proof=read(receipt)
            if file_hash(acoustic/'card-cache.json')!=proof['cache_sha256']:
                raise ValueError('mixed cache changed')
            view=read(acoustic/'card-cache.json')
            training=read(entry/'training.json')
            if (view['layout_plan_sha256']!=file_hash(entry/'card-plan.json') or
                view['source_identity']!=training['release_sha256'] or
                view['full_verified']!=training['batch_config']['source_queries'] or
                not (acoustic/'synthetic-cache').is_dir()):
                raise ValueError('mixed layout, release or population changed')
            commit_json(root/'receipt.json',dict(proof,reused_receipt=c['prepared_mixed']['receipt_sha256']))
        elif c.get('mixed'):
            from multimodal_jev.stage1_mixed_cache import build
            build(c,entry,root,init/'policy')
        elif reuse:
            from multimodal_jev.context_policy_artifacts import read_execution_bundle
            if (file_hash(init/'receipt.json') != reuse['receipt_sha256']
                    or any(file_hash(entry/k) != digest for k,digest in reuse['input_pins'].items())):
                raise ValueError('prepared inputs differ from the frozen release')
            proof = read(init/'receipt.json')
            if (file_hash(init/'card-cache.json') != proof['cache_sha256']
                    or read(init/'card-cache.json')['layout_plan_sha256'] != file_hash(entry/'card-plan.json')):
                raise ValueError('prepared cache differs')
            read_execution_bundle(init/'policy',expected_policy_id=proof['policy_id'],
                expected_contract=read(init/'policy/MANIFEST.json')['contract'],
                expected_base_id=read(entry/'training.json')['base_id'])
            commit_json(root/'receipt.json',dict(proof,reused_receipt=reuse['receipt_sha256']))
        else:
            prepare(c,root)
        return read(root/'receipt.json')
    proof = read(work/'prepare/receipt.json')
    parent = read(entry/'training.json')
    manifest = read(init/'policy/MANIFEST.json')
    design = read(entry/'design.json')
    budget = design['stage1']['total_updates']
    finite = role == 'finite'
    if not finite and read(work/'finite/receipt.json')['updates'] != c['finite_updates']:
        raise ValueError('two finite updates required')
    parent.update(source_root=str(entry/'source'),source_policy=str(init/'policy'),source_policy_id=proof['policy_id'],
        source_contract=manifest['contract'],stage_lora_transition=bind_design(design,manifest['contract']),
        stage1_layout={k:dict(path=str(entry/name),sha256=file_hash(entry/name))
                       for k,name in [('design','design.json'),('plan','card-plan.json')]},
        layout_cache=dict(path=str((acoustic if c.get('mixed') else init)/'card-cache.json'),sha256=proof['cache_sha256']),
        stop_update=c['finite_updates'] if finite else budget,engineering_prefix=finite,training_dispatch_allowed=not finite,
        checkpoint_interval=c['finite_updates'] if finite else c['checkpoint_interval'],evaluation_steps=c['evaluation_steps'],
        reference_endpoint=budget,selection_rule='predeclared_final_endpoint',parity_passes=c['parity_passes'] if finite else 0,
        parity_probability_atol=c['parity_probability_atol'])
    if c.get('mixed'):
        parent['mixed'].update(entry=str(entry),cache=str(acoustic/'synthetic-cache'))
    if 'execution_microbatch' in c:
        micro=c['execution_microbatch']
        if type(micro)!=int or micro<1 or parent['batch_config']['per_dp_replica_batch']%micro:
            raise ValueError('microbatch must divide the effective replica batch')
        parent['batch_config']['microbatch']=parent['microbatch']=micro
    commit_json(root/'config.json',parent)
    subprocess.run([sys.executable,'-m','torch.distributed.run','--nnodes=1','--node-rank=0',
        '--master-addr='+c['master_addr'],'--master-port='+str(c['master_port']),'--max-restarts=0',
        '--nproc-per-node='+str(len(parent['physical_devices'])),str(entry/'source/tools/jev/train_card_stage1.py'),
        '--config',str(root/'config.json'),'--output',str(root/'run')],check=True)
    result = read(root/'run/result.json')
    if result['passed'] is not True or result['stop_update']!=parent['stop_update'] or result['engineering_prefix']!=finite:
        raise ValueError('stage1 terminal differs')
    for rank in range(len(parent['physical_devices'])):
        rank_root=root/'run'/f'rank-{rank}'
        if finite and read(rank_root/'inference-parity.json')['compared']!=c['parity_passes']*parent['microbatch']:
            raise ValueError('finite inference parity incomplete')
        for step in range(1,parent['stop_update']+1):
            record=read(rank_root/f'update-{step:04d}.json')
            if record['update']!=step or not math.isfinite(record['metrics']['actor/pg_loss']):
                raise ValueError('missing optimizer update')
    receipt=dict(updates=result['stop_update'],engineering_prefix=finite,config_sha256=file_hash(root/'config.json'))
    commit_json(root/'receipt.json',receipt)
    return receipt
