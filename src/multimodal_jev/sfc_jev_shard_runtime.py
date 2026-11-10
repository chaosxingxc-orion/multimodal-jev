"""Frozen parent-only SFC-Jev shards bound to full CPU context evidence."""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc_jev_runtime import INSTRUCTION
from .sfc_jev_shards import load_shard
from .sfc_native_full import RUNTIME


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def select_inputs(cfg):
    if (any(cfg.get(k)!=v for k,v in RUNTIME.items()) or cfg.get('adapter') is not None
            or cfg.get('prompt_id')!='parent'
            or cfg.get('instruction_sha256')!=hashlib.sha256(INSTRUCTION.encode()).hexdigest()):
        raise ValueError('frozen parent-only Jev runtime required')
    path=Path(cfg['shard_plan'])
    if file_hash(path)!=cfg['shard_plan_sha256']:raise ValueError('registered plan changed')
    plan=read(path)
    if cfg['split']!=plan['split']:raise ValueError('registered split differs')
    rows=load_shard(plan,cfg['shard_id'])
    root=Path(cfg['context_audit']);run=root/'run';result=read(run/'result.json');identity=read(run/'identity.json')
    acceptance=read(cfg['context_acceptance'])
    if (file_hash(run/'result.json')!=cfg['context_result_sha256']
            or file_hash(cfg['context_acceptance'])!=cfg['context_acceptance_sha256']
            or acceptance['result_sha256']!=cfg['context_result_sha256']
            or acceptance['container_id']!=cfg['context_container_id']
            or acceptance['all_inputs_fit'] is not True
            or result['status']!='passed' or result['all_inputs_fit'] is not True
            or read(run/'terminal.json')['exit_code']!=0
            or file_hash(run/'identity.json')!=result['identity_sha256']
            or identity['model']!=cfg['audio_model'] or identity['release']!=plan['release']
            or identity['manifest_sha256']!=plan['manifest_sha256'] or identity['targets_read'] is not False
            or identity['context_limit']!=4096 or identity['output_budget']!=1024):
        raise ValueError('successful independently accepted complete parent context audit required')
    for name,value in identity['source_files'].items():
        if file_hash(root/'source'/name)!=value:raise ValueError('context audit source changed')
    for name in ('src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_audio.py'):
        if file_hash(Path(__file__).parents[2]/name)!=identity['source_files'][name]:
            raise ValueError('parent adapter or audio processing differs from context audit')
    for name,value in identity['processor_assets'].items():
        if file_hash(Path(cfg['audio_model'])/name)!=value:raise ValueError('processor asset changed')
    path=run/f"{plan['split']}.context.jsonl"
    if file_hash(path)!=result['splits'][plan['split']]['records_sha256']:
        raise ValueError('context records changed')
    wanted={(r['id'],d['id']) for r in rows for d in r['input']['decisions']};checks={}
    with path.open() as source:
        for line in source:
            record=json.loads(line);key=(record['id'],record['decision_id'])
            if key not in wanted:continue
            if key in checks:raise ValueError('duplicate context decision')
            if (record['context_overflow'] is not False or record['input_tokens']+1024>4096
                    or record['expected_frames']!=record['observed_frames']):
                raise ValueError('complete fitting parent context required')
            checks[key]=record
    if set(checks)!=wanted:raise ValueError('every shard decision needs CPU evidence')
    return rows,plan,checks


def check_probe(cfg):
    """Verify local TRAIN78 records; launcher/verifier also inspect exact Docker ID."""
    root=Path(cfg['jev_probe']);run=root/'run'
    if file_hash(root/'acceptance.json')!=cfg['jev_probe_acceptance_sha256']:
        raise ValueError('registered Jev TRAIN acceptance changed')
    accepted=read(root/'acceptance.json');identity=read(run/'identity.json');result=read(run/'result.json')
    prior=read(root/'config.json');prep=read(root/'preparation.json');launch=read(root/'launch.json')
    if (accepted['bundles']!=6 or accepted['decisions']!=78 or accepted['source_groups']!=4
            or accepted['weight_updates']!=0 or file_hash(root/'score/metrics.json')!=accepted['metrics_sha256']
            or read(root/'evaluation-terminal.json')['exit_code']!=0 or read(run/'terminal.json')['exit_code']!=0
            or launch['container_id']!=cfg['jev_probe_container_id'] or launch['image']!=cfg['image']
            or prep['image']!=cfg['image'] or prep['config_sha256']!=file_hash(root/'config.json')
            or result['status']!='passed' or result['weight_updates']!=0 or result['model_files_unchanged'] is not True
            or result['bundles']!=6 or result['decisions']!=78 or result['predictions']!=78
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']
            or identity['config']!=prior or prior['split']!='train_id'
            or any(prior.get(k)!=v for k,v in RUNTIME.items())
            or any(prior[k]!=cfg[k] for k in ('versions','audio_model'))):
        raise ValueError('complete successful frozen Jev TRAIN78 required')
    for name,value in identity['source_files'].items():
        if file_hash(root/'source'/name)!=value or prep['inference_files'].get(name)!=value:
            raise ValueError('TRAIN probe source changed')
    for name in ('src/multimodal_jev/sfc_jev_runtime.py','src/multimodal_jev/sfc_audio.py'):
        if file_hash(Path(__file__).parents[2]/name)!=identity['source_files'][name]:
            raise ValueError('shard adapter differs from accepted TRAIN')
    return identity


def check_predictions(rows,checks,predictions):
    keys=[(r['id'],d['id']) for r in rows for d in r['input']['decisions']]
    if [(p['example_id'],p['decision_id']) for p in predictions]!=keys:
        raise ValueError('every ordered shard decision required exactly once')
    by_id={r['id']:r for r in rows}
    for p in predictions:
        key=(p['example_id'],p['decision_id']);row=by_id[key[0]];check=checks[key]
        if (p['request_sha256']!=digest(row) or p['audio_sha256']!=row['audio_sha256']
                or p['split']!=row['split'] or p['source_group']!=row['source_group']
                or p['hf_vllm_prompt_ids_equal'] is not True
                or any(p[k]!=check[k] for k in ('input_tokens','input_ids_sha256','prompt_sha256','expected_frames','observed_frames'))
                or type(p['raw_text']) is not str or p['finish_reason'] not in ('stop','length')
                or type(p['length_limited']) is not bool or p['length_limited']!=(p['finish_reason']=='length')
                or type(p['token_ids']) is not list or len(p['token_ids'])>1024
                or any(type(t) is not int or t<0 for t in p['token_ids'])):
            raise ValueError('shard prediction binding, engine parity or generation budget changed')
