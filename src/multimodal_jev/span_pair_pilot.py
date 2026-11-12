"""Independent one-step engineering protocol; no extension of old study methods."""
import json
from pathlib import Path
from .baseline import file_hash
from .span_transfer_runtime import branch_protocol,step_rows
from .span_pair_credit import learning_pairs


def pilot_rows(config):
    protocol_path=Path(config['protocol'])
    if config.get('pins',{}).get(str(protocol_path))!=file_hash(protocol_path):
        raise ValueError('pinned pair pilot protocol required')
    protocol=json.loads(protocol_path.read_text())
    for path,digest in protocol['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pair pilot input changed')
    if (protocol.get('schema')!='jev-pair-engineering-pilot-v1'
            or protocol.get('algorithm')!='exact-complement-pair-marginal-ppo-v1'
            or protocol.get('pair_weights')!=[0.,.5]
            or config['pair_weight'] not in protocol['pair_weights']
            or protocol.get('num_updates')!=1 or protocol.get('auxiliary_weight')!=0.
            or protocol.get('clip_ratio')!=.2 or protocol.get('optimizer_start')!='fresh'
            or protocol.get('lr')!=1e-4 or protocol.get('seed')!=17
            or protocol.get('arm')!='structured' or protocol.get('batch_index')!=0):
        raise ValueError('pair pilot contract differs')
    source=Path(protocol['source_step'])
    if str(source) not in protocol['pins']:raise ValueError('pinned source batch required')
    batch=json.loads(source.read_text());study=branch_protocol(batch);rows=step_rows(batch)
    for key in ('seed','arm','batch_index','lr'):
        if batch[key]!=protocol[key]:raise ValueError('source batch differs')
    if (batch['adapter_id']!=study['initializer']['adapter_id']
            or protocol['initial_adapter_id']!=batch['adapter_id']
            or protocol['base_id']!=batch['base_id']
            or protocol['runtime_admission']!=batch['runtime_admission']):
        raise ValueError('initial policy or runtime differs')
    pairs=learning_pairs(rows,study['eligible_training_request_ids'])
    if pairs.shape!=(4,2):raise ValueError('four reviewed pairs required')
    return protocol,batch,rows,pairs
