"""Label-free target-width HF versus actual macro-service parity.

Run HF and macro phases in separate isolated processes. This client does not
launch containers or certify their provenance; no target score is computed.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_logits_wire import validate_logits_record
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_training import content_hash
from tools.reports.context_transfer_macro_inputs import load,audited_encoder

WIDTHS={'FSC_native31':31,'MINDS_native14':14,'SLURP_clean_TEST60':60}


def panel(admitted):
    """Deterministic input-only panel: both length axes and public strata.

    FSC supplies speakers. Current MINDS/SLURP label-free adapters expose no
    language/speaker strata; do not open their gold inventories to recover them.
    Candidate labels, probabilities and errors are never read.
    """
    dataset=admitted['dataset'];population=admitted['population']
    receipts=admitted['input_receipts'];rows=admitted['source_rows']
    candidates=admitted['candidate_ids']
    if (dataset not in WIDTHS or len(candidates)!=WIDTHS[dataset]
            or len(set(candidates))!=len(candidates) or not population
            or len(rows)!=len(population) or len(receipts)!=len(population)
            or len({r['id'] for r in population})!=len(population)):
        raise ValueError('complete aligned native target population required')
    chosen=set();costs=[];strata={}
    for i,(asset,receipt,row) in enumerate(zip(population,receipts,rows,strict=True)):
        if (set(asset)!={'id','audio_sha256','request_sha256'}
                or any(receipt.get(k)!=v for k,v in asset.items()) or row['id']!=asset['id']
                or receipt['candidate_ids']!=candidates):
            raise ValueError('independent label-free input binding differs')
        positions=receipt['positions'];inputs=receipt['inputs']
        text=inputs['input_ids']['shape'];audio=inputs['input_features']['shape']
        if (len(text)!=2 or text[0]!=1 or not 0<text[1]<=4096
                or not audio or type(audio[-1]) is not int or audio[-1]<=0
                or len(positions)!=len(candidates)
                or any(type(p) is not int or not 0<=p<text[1] for p in positions)
                or positions!=sorted(set(positions))):
            raise ValueError('bounded input lengths and complete ordered markers required')
        costs.append((text[1],audio[-1]))
        key=None
        if dataset=='FSC_native31':key=row.get('speaker_id')
        if dataset=='FSC_native31':
            if not isinstance(key,str) or not key:raise ValueError('public target stratum missing')
            strata.setdefault(key,[]).append(i)
    # Select by ID within strata, not caller order. Always return population order.
    for indices in strata.values():chosen.add(min(indices,key=lambda i:population[i]['id']))
    for axis in (0,1):
        ordered=sorted(range(len(population)),key=lambda i:(costs[i][axis],population[i]['id']))
        chosen.update(ordered[:16]);chosen.update(ordered[-16:])
    # Keep the existing MINDS/SLURP 256-input admission panel as a subset.
    # The extra audio-length extremes extend coverage, never replace it.
    if dataset in ('MINDS_native14','SLURP_clean_TEST60'):
        ordered=sorted(range(len(population)),key=lambda i:(costs[i][0],population[i]['id']))
        original=set(ordered[:16]+ordered[-16:])
        for i in sorted(range(len(population)),key=lambda i:(content_hash(population[i]['id']),population[i]['id'])):
            if len(original)>=min(256,len(population)):break
            original.add(i)
        chosen.update(original)
    indices=sorted(chosen)
    return dict(schema='jev-target-width-parity-panel-v1',dataset=dataset,
        candidate_ids=candidates,processor_sha256=admitted['processor_sha256'],
        population_sha256=digest(population),input_receipts_sha256=digest(receipts),
        indices=indices,assets=[population[i] for i in indices],
        input_receipts=[receipts[i] for i in indices],
        selection='text/audio extremes and public strata; MINDS/SLURP retain original256 length/hash panel',
        covered_strata=sorted(strata),stratum_field='speaker_id' if dataset=='FSC_native31' else None,
        language_coverage_verified=False,target_labels_used=False)


def compare(reference,records,expected,*,policy_id):
    """Require actual logits as well as probabilities; saturated values can hide errors."""
    if (reference.get('schema')!='jev-target-width-hf-reference-v1'
            or reference.get('panel_sha256')!=digest(expected) or reference.get('policy_id')!=policy_id
            or len(reference['records'])!=len(expected['assets']) or len(records)!=len(expected['assets'])):
        raise ValueError('complete same-policy independent HF panel required')
    max_prob=0.;max_logit=0.;flips=0;top1=0
    for asset,receipt,hf,vl in zip(expected['assets'],expected['input_receipts'],reference['records'],records,strict=True):
        identity=dict(policy_id=policy_id,positions=receipt['positions'],inputs=receipt['inputs'])
        if (hf['id']!=asset['id'] or vl['id']!=asset['id'] or hf['request_identity']!=identity
                or vl['request_identity']!=identity or vl['request_id']!=digest(identity)
                or vl['policy_id']!=policy_id):
            raise ValueError('HF/macro audio input or policy identity differs')
        validate_logits_record(vl)
        width=len(expected['candidate_ids'])
        if any(len(r[k])!=width for r in (hf,vl) for k in ('prob','logits')):
            raise ValueError('complete target decision width required')
        if any(type(x) not in (float,int) or not math.isfinite(x) for k in ('prob','logits') for x in hf[k]):
            raise ValueError('finite actual HF outputs required')
        p=torch.tensor(hf['prob'],dtype=torch.float32);z=torch.tensor(hf['logits'],dtype=torch.float32)
        if ((p<=0)|(p>=1)).any() or (z.sigmoid()-p).abs().max().item()>2e-7:
            raise ValueError('HF actual logits and probabilities incompatible')
        max_prob=max(max_prob,max(abs(a-b) for a,b in zip(hf['prob'],vl['prob'],strict=True)))
        max_logit=max(max_logit,max(abs(a-b) for a,b in zip(hf['logits'],vl['logits'],strict=True)))
        flips+=sum((a>=.5)!=(b>=.5) for a,b in zip(hf['prob'],vl['prob'],strict=True))
        top1+=max(range(width),key=lambda i:hf['prob'][i])!=max(range(width),key=lambda i:vl['prob'][i])
    return dict(dataset=expected['dataset'],policy_id=policy_id,panel_sha256=digest(expected),
        queries=len(records),decisions=len(expected['candidate_ids']),probability_max_abs=max_prob,
        logit_max_abs=max_logit,threshold_flips=flips,top1_changes=top1,
        exact=max_prob==0 and max_logit==0,full_population_parity=False,
        runtime_provenance_verified=False,target_scored=False)


def load_reference_model(config,bundle,output):
    import torch_npu
    devices=config['physical_devices']
    if config.get('process_device_scope') is not None:
        from tools.jev.context_configurable_device_scope import validate_tp1
        validate_tp1(config)
    elif len(devices)!=1 or sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=['davinci'+str(devices[0])]:
        raise ValueError('one isolated HF NPU required')
    torch.set_num_threads(2);torch.npu.set_device(0);torch.manual_seed(317);torch.npu.manual_seed_all(317)
    return load_native_reference(config,bundle,output,torch.device('npu',0))


def load_native_reference(config,bundle,output,device):
    from multimodal_jev.asr import load_runtime
    from multimodal_jev.sft import initialize_adapter
    from multimodal_jev.context_model_profile import from_hf,build_head
    from multimodal_jev.context_decision_model import ContextDecisionModel
    from multimodal_jev.context_policy_snapshot import load_policy
    from multimodal_jev.context_runtime_profile import PROFILE,install_actor

    profile=from_hf(config['model'])
    contract=bundle['contract']
    if (set(profile.lora_targets())!=set(contract['lora'])
            or any(contract['branch'].get(k)!=v for k,v in profile.decision_head().items())):
        raise ValueError('HF endpoint decision head or LoRA targets differ from model geometry')
    ranks_alpha={(item['rank'],item['alpha']) for item in contract['lora'].values()}
    if len(ranks_alpha)!=1:
        raise ValueError('HF endpoint requires uniform LoRA rank and alpha')
    rank,alpha=next(iter(ranks_alpha))
    _,base=load_runtime(config['model'],device,'bfloat16')
    adapted=initialize_adapter(base,dict(lora_rank=rank,lora_alpha=alpha,lora_execution_dtype='bfloat16')).eval()
    head=build_head(profile).to(device=device,dtype=torch.bfloat16).eval()
    model=ContextDecisionModel(adapted.get_base_model().model,head,bundle['contract']['token']['id']).eval()
    load_policy(config['snapshot'],model,expected_policy_id=config['policy_id'],
        expected_contract=config['contract'],expected_base_id=config['base_id'])
    handle,runtime=install_actor(adapted,model,PROFILE)
    commit_json(output/'hf-runtime.json',runtime)
    return model,device,base,handle


def hf_reference(config,admitted,selected,bundle,encode,output):
    model,device,base,handle=load_reference_model(config,bundle,output)
    records=[]
    for asset,receipt in zip(selected['assets'],selected['input_receipts'],strict=True):
        row,actual=encode(asset)
        if actual!=receipt:raise ValueError('actual HF input evidence differs')
        with torch.no_grad():
            logits=model({k:v.to(device) for k,v in row['inputs'].items()},
                row['positions'].to(device),return_logits=True).float()[0]
            probability=logits.sigmoid()
        records.append(dict(id=asset['id'],request_identity=dict(policy_id=config['policy_id'],
            positions=receipt['positions'],inputs=receipt['inputs']),
            logits=logits.cpu().tolist(),prob=probability.cpu().tolist()))
    return dict(schema='jev-target-width-hf-reference-v1',policy_id=config['policy_id'],
        panel_sha256=digest(selected),records=records)


def run(config,provenance,phase):
    if phase not in ('prepare','hf','macro'):raise ValueError('explicit parity phase required')
    for path,h in config.get('pins',{}).items():
        if file_hash(path)!=h:raise ValueError('pinned dependency changed: '+path)
    if 'portable_parity_inputs' in config:
        from tools.reports.context_transfer_parity_inputs import load as load_portable
        admitted=load_portable(config['portable_parity_inputs']);selected=admitted['selected_panel']
        if admitted['dataset']!=config['dataset']:raise ValueError('portable dataset differs')
    else:
        admitted=load(config);selected=panel(admitted)
    out=Path(config['output']);check_paths({'parity':out});out.mkdir(parents=True,exist_ok=False)
    commit_json(out/'panel.json',selected);commit_json(out/'provenance.json',provenance)
    if phase=='prepare':
        report=dict(status='label_free_panel_prepared',panel_sha256=digest(selected),
            queries=len(selected['assets']),dataset=config['dataset'],npu_executed=False)
        commit_json(out/'result.json',report);return report
    pool=None
    try:
        from multimodal_jev.policy_identity import checkpoint_identity
        from multimodal_jev.context_runtime_profile import PROFILE
        if checkpoint_identity(config['model'])[0]!=config['base_id']:raise ValueError('base identity differs')
        bundle=read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
            expected_contract=config['contract'],expected_base_id=config['base_id'])
        if (bundle['contract']['runtime_profile']!=PROFILE
                or bundle['contract']['request_version']!='jev-context-exclusive_event-v1'
                or bundle['contract']['token']['registration_status']!='admitted'):
            raise ValueError('admitted context policy required')
        verify_native_adapter(config['adapter'],bundle)
        encode=audited_encoder(config,admitted,snapshot=config['snapshot'],
            decision_token_id=bundle['contract']['token']['id'],audio_token_id=config['audio_token_id'])
        if phase=='hf':
            report=hf_reference(config,admitted,selected,bundle,encode,out)
            commit_json(out/'reference.json',report)
        else:
            from tools.reports.evaluate_context_transfer_macro import read_bound
            from tools.reports.context_macro_client import connect_pool
            from tools.reports.context_macro_evaluation import evaluate
            from tools.reports.audit_context_macro_evaluation import audit
            reference=read_bound(config['hf_reference'])
            # Validate the HF panel before contacting either inference service.
            if reference.get('panel_sha256')!=digest(selected) or reference.get('policy_id')!=config['policy_id']:
                raise ValueError('HF reference panel/policy differs')
            pool=connect_pool(config['macro_client'],bundle=bundle)
            evaluate(pool,selected['assets'],encode,candidate_ids=selected['candidate_ids'],
                policy_id=config['policy_id'],runtime_identity=config['macro_client']['runtime_identity'],
                audio_token_id=config['audio_token_id'],macro_queries=128,output=out/'inference')
            pool.close();pool=None
            proof=audit(out/'inference',Path(config['macro_client']['journal_root'])/config['macro_client']['attempt_id'],
                population=selected['assets'],candidate_ids=selected['candidate_ids'],
                input_receipts=selected['input_receipts'],policy_id=config['policy_id'],
                runtime_identity=config['macro_client']['runtime_identity'],macro_queries=128)
            commit_json(out/'macro-output-audit.json',proof)
            records=json.loads((out/'inference/predictions.json').read_text())['records']
            report=compare(reference,records,selected,policy_id=config['policy_id'])
            report.update(hf_reference=config['hf_reference'],
                macro_output_audit_sha256=file_hash(out/'macro-output-audit.json'))
            commit_json(out/'comparison.json',report)
            if not report['exact']:raise ValueError('target-width HF/macro parity failed')
        commit_json(out/'terminal.json',dict(exit_code=0));return report
    except BaseException as exc:
        commit_json(out/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    finally:
        if pool is not None:pool.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--phase',choices=['prepare','hf','macro'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance,a.phase),indent=2))
