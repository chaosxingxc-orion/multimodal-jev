"""Bind a verified input/runtime contract to unchanged Stage1 initialization.

An admission here covers the execution protocol, not task quality or all future
policy probabilities. New RL behavior must still pass per-update parity checks.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import torch

from .baseline import file_hash
from .context_policy_artifacts import read_execution_bundle
from .context_policy_snapshot import _hash
from .context_runtime_profile import PROFILE
from .context_token_contract import validate_processor_extension
from .local_checkpoint import sync_directory
from .local_runtime_storage import check_paths


def admit_initial_policy(source,verified,destination,*,base,parity,prefix,replay,token_assets):
    source,verified,destination=Path(source),Path(verified),Path(destination)
    a=json.loads((source/'MANIFEST.json').read_text());b=json.loads((verified/'MANIFEST.json').read_text())
    def read(path,manifest):
        return read_execution_bundle(path,expected_policy_id=manifest['policy_id'],
            expected_contract=manifest['contract'],expected_base_id=manifest['contract']['base_id'])
    initial,checked=read(source,a),read(verified,b)
    contract=checked['contract']
    if (contract['runtime_profile']!=PROFILE or contract['request_version']!='jev-context-exclusive_event-v1' or
            contract['token']!=dict(id=151706,text='<|jev_decision|>',row_trainable=False,registration_status='test_only')):
        raise ValueError('verified actual decision runtime contract required')
    for name in ('base_id','branch','token','output','lora'):
        if initial['contract'][name]!=contract[name]:raise ValueError('initialization execution semantics differ')
    processor=lambda m:{k:v for k,v in m['files'].items() if k.startswith('processor/')}
    if processor(a)!=processor(b) or not torch.equal(initial['tensors']['decision_token_row'],checked['tensors']['decision_token_row']):
        raise ValueError('verified processor or frozen marker differs')
    validate_processor_extension(base,source/'processor')
    for report in (parity,prefix,replay):
        if report['state']['ExitCode']!=0 or report['state']['OOMKilled']:
            raise ValueError('successful independent execution evidence required')
    rows=parity['independent_probability_checks']
    if (parity['joint_policy_id']!=b['policy_id'] or len(rows)!=256 or len({r['id'] for r in rows})!=256 or
            any(r['probability_max_error']!=0 or r['threshold_half_disagreements']!=0 for r in rows)):
        raise ValueError('complete verified-policy probability parity required')
    if (not prefix['result']['passed'] or len(prefix['rank_updates'])!=8 or
            replay['fresh_process_replay']!=dict(exact_four_rank_masters_adam=True,exact_128_bf16_execution=True)):
        raise ValueError('actual full-loop and fresh-process recovery evidence required')
    from .span_request_identity import tensor_identity
    if (token_assets['source_policy_id']!=a['policy_id'] or not token_assets['original_tokenizer_preserved'] or
            token_assets['embedding_initialization']['exact'] is not True or
            token_assets['embedding_initialization']['marker']!=tensor_identity(initial['tensors']['decision_token_row'])):
        raise ValueError('initial marker must match independently verified base-row initialization')
    contract=copy.deepcopy(contract);contract['token']['registration_status']='admitted'
    check_paths({'admitted_policy':destination})
    if destination.exists():raise FileExistsError(destination)
    stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        # Copy exact execution bytes, including zero-init LoRA. No diagnostic
        # training weights may replace the Stage1 research initialization.
        shutil.copytree(source/'processor',stage/'processor')
        shutil.copyfile(source/'execution.pt',stage/'execution.pt')
        files={str(p.relative_to(stage)):file_hash(p) for p in stage.rglob('*') if p.is_file()}
        if files!=a['files']:raise ValueError('initial policy bytes changed')
        payload=dict(schema='context-policy-execution-v1',contract=contract,files=files);identity=_hash(payload)
        (stage/'MANIFEST.json').write_text(json.dumps(dict(payload,policy_id=identity),indent=2)+'\n')
        fresh=read_execution_bundle(stage,expected_policy_id=identity,expected_contract=contract,expected_base_id=contract['base_id'])
        if any(not torch.equal(v,fresh['tensors'][k]) for k,v in initial['tensors'].items()):
            raise ValueError('initialization changed on admission')
        for p in stage.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fchmod(f.fileno(),0o644);os.fsync(f.fileno())
        for p in sorted((p for p in stage.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):sync_directory(p)
        sync_directory(stage);os.rename(stage,destination);sync_directory(destination.parent)
        return dict(policy_id=identity,source_policy_id=a['policy_id'],verified_policy_id=b['policy_id'],
            contract=contract,unchanged_initialization_files=files,
            scope='Stage1 input/runtime/token protocol admitted with unchanged initialization; future policy parity, RL activation and efficacy remain separate.')
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True);raise
