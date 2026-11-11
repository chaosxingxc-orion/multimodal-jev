"""Pinned transfer-study branch, step cursor and runtime coverage contracts."""
import json
from pathlib import Path
from .baseline import file_hash
from .span_rule_population import select_rule_rows,LEARNING_PROBE
from .span_rule_transfer import semantic_rule_group

POPULATION='transfer-version-gate-v1'
METHODS={'sft':'sft','rl':'joint-grpo','conditional':'exact-conditional-marginal-ppo'}


def branch_protocol(config):
    path=config['transfer_protocol']
    if config.get('pins',{}).get(path)!=file_hash(path):raise ValueError('pinned transfer protocol required')
    p=json.loads(Path(path).read_text())
    if (p['schema']!='jev-span-rule-transfer-protocol-v1' or config['release']!=p['learning_release']
            or config['seed'] not in p['seeds'] or config['arm'] not in p['training_arms']
            or METHODS.get(config['method']) not in p['methods']
            or config['auxiliary_weight'] not in p['auxiliary_weights']
            or (config['arm']=='final_only' and config['auxiliary_weight']!=0)
            or config['runtime_profile']!=p['runtime_profile']
            or config['lr']!=p['optimizer']['lr']
            or config['samples_per_request']!=p['joint_grpo_samples_per_request']):
        raise ValueError('branch differs from transfer protocol')
    for path,digest in p['pins'].items():
        if file_hash(path)!=digest:raise ValueError('transfer source release changed')
    return p


def learning_rows(config):
    release=Path(config['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    if manifest['schema']!='jev-span-balanced-rule-controls-v1':raise ValueError('learning rule release required')
    for name,digest in manifest['files'].items():
        if file_hash(release/name)!=digest:raise ValueError('learning release changed')
    rows=[json.loads(s) for s in (release/'learning.jsonl').read_text().splitlines()]
    if len({r['id'] for r in rows})!=len(rows) or any(r['split']!='learning' for r in rows):
        raise ValueError('unique learning population required')
    return {r['id']:r for r in rows}


def step_rows(config):
    p=branch_protocol(config);i=config['batch_index']
    if type(i) is not int or not 0<=i<p['updates_per_branch']:raise ValueError('invalid training cursor')
    ids=p['schedules'][str(config['seed'])][i]['arms'][config['arm']]
    rows=learning_rows(config);selected=[rows[k] for k in ids]
    eligible=set(p['eligible_training_request_ids'])
    if (ids!=config['request_ids'] or not set(ids)<=eligible or len(selected)!=8
            or len({r['provenance']['source_group'] for r in selected})!=4
            or any(r['arm']!=config['arm'] or not all(v is True for v in r['supervised'])
                   or semantic_rule_group(r['provenance']['rule'])==p['heldout_semantic_group'] for r in selected)
            or any(sum(r['targets'][j] is True for r in selected)!=4 for j in range(len(selected[0]['targets'])))):
        raise ValueError('step differs from reviewed balanced non-heldout batch')
    return selected


def version_rows(config):
    """Every version gates fixed 216 probes plus previous and next batch inputs."""
    p=branch_protocol(config);version=config['version_index']
    if type(version) is not int or not 0<=version<=p['updates_per_branch']:
        raise ValueError('invalid runtime version index')
    selected=select_rule_rows(Path(config['release']),LEARNING_PROBE);seen={r['id'] for r in selected}
    for index in (version-1,version):
        if 0<=index<p['updates_per_branch']:
            step={**config,'batch_index':index,'request_ids':p['schedules'][str(config['seed'])][index]['arms'][config['arm']]}
            for r in step_rows(step):
                if r['id'] not in seen:selected.append(r);seen.add(r['id'])
    if config['request_ids']!=[r['id'] for r in selected]:raise ValueError('runtime coverage differs from fixed probe/adjacent steps')
    return selected


def cursor(config,next_index):
    return dict(schema='jev-rule-transfer-cursor-v1',protocol_sha256=file_hash(config['transfer_protocol']),
        method=config['method'],arm=config['arm'],auxiliary_weight=config['auxiliary_weight'],
        seed=config['seed'],next_batch_index=next_index)


def validate_parent(config,receipt):
    index=config['batch_index'];p=branch_protocol(config)
    if index==0:
        if receipt is not None or config.get('initial_optimizer_checkpoint'):
            raise ValueError('transfer starts with fresh optimizer')
        if config['adapter_id']!=p['initializer']['adapter_id']:raise ValueError('fact-only initializer required')
    elif (receipt is None or receipt['training_state']!=cursor(config,index)
            or receipt['completed_updates']!=index or receipt['adapter_id']!=config['adapter_id']):
        raise ValueError('resume cursor, branch or adapter differs')
