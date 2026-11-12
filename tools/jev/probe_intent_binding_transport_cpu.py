#!/usr/bin/env python3
"""Real audio encoding and four-rank verl transport; synthetic policies only."""
import argparse
import copy
import json
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multimodal_jev.baseline import atomic_json,file_hash


def worker(rank,output):
    from multimodal_jev.span_finite_actor import make_global_batch,admit_batch
    from probe_span_native_actor_dp_cpu import config
    output=Path(output);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=(output/'gloo-init').resolve().as_uri(),
        rank=rank,world_size=4,timeout=timedelta(seconds=120))
    try:
        cases=[]
        for arm in ('D','Q','P'):
            items=torch.load(output/(arm+'.pt'),weights_only=False) if rank==0 else None
            chunks=list(make_global_batch(items).split(8)) if rank==0 else None
            incoming=[None];dist.scatter_object_list(incoming,chunks,src=0);batch=incoming[0]
            cfg=config(1,'synthetic-binding-transport-only');cfg.ppo_mini_batch_size=8
            admission=admit_batch(batch,cfg)
            # Independently derive expected row credit from external transformed gold.
            targets=json.loads((output/(arm+'-targets.json')).read_text())[rank*8:(rank+1)*8]
            for payload,target in zip(batch.non_tensor_batch['multi_modal_inputs'],targets,strict=True):
                assert payload['effective_weight']==target['request_weight']
                for j,(old,adv) in enumerate(zip(payload['behavior'],payload['advantages'],strict=True)):
                    expected=-old.exp()[0,target['targets'][j]].expand_as(old).clone()
                    expected[0,target['targets'][j]]+=1
                    expected*=target['span_weights'][j]
                    torch.testing.assert_close(adv,expected,atol=1e-7,rtol=1e-6)
                assert 'targets' not in payload['inputs'] and 'target_values' not in payload['inputs']
            bad=copy.deepcopy(batch)
            if rank==2:bad.batch['advantages'][0,0]+=.25
            try:admit_batch(bad,cfg)
            except ValueError:rejected=True
            else:raise AssertionError('altered transported credit accepted')
            cases.append(dict(arm=arm,admission=admission,altered_credit_rejected=rejected))
        atomic_json(output/f'rank-{rank}.json',dict(passed=True,cases=cases))
    finally:dist.destroy_process_group()


def run(cfg,output):
    from transformers import Qwen3ASRProcessor
    from multimodal_jev.intent_binding_inputs import load,select,encode,prepare_item
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU only')
    for path,digest in cfg['pins'].items():
        if file_hash(path)!=digest:raise ValueError('pinned input changed: '+path)
    processor=Qwen3ASRProcessor.from_pretrained(cfg['model'],local_files_only=True)
    evidence=json.loads(Path(cfg['processor_result']).read_text())
    if not evidence['passed'] or len(evidence['records'])!=712:raise ValueError('complete processor evidence required')
    records={r['id']:r for r in evidence['records']}
    policy='synthetic-binding-transport-only';receipts={};cached={}
    for arm in ('D','Q','P'):
        rows,admission=load(cfg['release'],cfg['manifest_sha256'],arm)
        selected,selection=select(rows,317,0);prepared=[]
        for row in selected:
            # P and Q share inputs. Cache only synthetic policy evidence, never labels.
            if row['gate_id'] not in cached:
                encoded,identity=encode(row,processor,policy)
                ref=records[row['gate_id']]
                if identity['request']!=ref['request'] or identity['positions']!=ref['positions']:
                    raise ValueError('transformed processor request changed')
                for key in ('input_ids','input_features','input_features_mask'):
                    if tensor_identity(encoded['inputs'][key])!=ref[key]:raise ValueError('processor tensors changed')
                # Nonuniform deterministic probabilities, explicitly not a speech-model gate.
                probs=[torch.log_softmax(torch.linspace(-1,1,len(c)),0).tolist() for c in encoded['candidates']]
                cached[row['gate_id']]=dict(id=row['gate_id'],request_id=encoded['request_id'],
                    request_identity=identity,candidate_logprobs=probs)
            record=cached[row['gate_id']]
            item,receipt=prepare_item(row,processor,policy,record,copy.deepcopy(record))
            prepared.append(item)
            # Reject native positional targets on a rebound semantic input.
            changed=copy.deepcopy(row);changed['target']['targets'][-1]=(changed['target']['targets'][-1]+1)%60
            try:prepare_item(changed,processor,policy,record,copy.deepcopy(record))
            except ValueError:pass
            else:raise AssertionError('incorrect semantic target accepted')
        assert len({x['request_id'] for x in prepared})==32
        torch.save(prepared,output/(arm+'.pt'))
        atomic_json(output/(arm+'-targets.json'),[r['target'] for r in selected])
        receipts[arm]=dict(selection=selection,admission=admission,
            request_ids=[x['request_id'] for x in prepared],
            input_tokens=[x['inputs']['input_ids'].shape[-1] for x in prepared])
        atomic_json(output/'progress.json',dict(completed_arm=arm))
    assert receipts['Q']['request_ids']==receipts['P']['request_ids']
    mp.spawn(worker,args=(str(output),),nprocs=4,join=True)
    ranks=[json.loads((output/f'rank-{i}.json').read_text()) for i in range(4)]
    result=dict(passed=True,receipts=receipts,ranks=ranks,
        scope='Real processor/audio/remapped labels, native verl DataProto and four Gloo ranks; '
              'synthetic nonuniform probabilities. No speech model forward, optimizer update, '
              'HF/vLLM policy gate, NPU/BF16 or training admission.')
    atomic_json(output/'result.json',result)
    for arm in ('D','Q','P'):(output/(arm+'.pt')).unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);torch.set_num_threads(2)
    try:
        run(json.loads(a.config.read_text()),a.output);atomic_json(a.output/'terminal.json',dict(exit_code=0))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
