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


def verify_pack_inputs(previous, current, old_root, new_root, *, rebind_plan):
    for key in ('population', 'policy_id'):
        if previous[key] != current[key]:
            raise ValueError('previous acoustic pack population or policy differs')
    for name, key, field in [('card-cache.json', 'cache_sha256', 'layout_plan_sha256'),
                             ('single.json', 'single_sha256', 'plan')]:
        if previous[key] == current[key]:
            continue
        if not rebind_plan:
            raise ValueError('previous prepared audio or cards differ')
        old, new = old_root/name, new_root/name
        if file_hash(old) != previous[key] or file_hash(new) != current[key]:
            raise ValueError('prepared input bytes differ from receipt')
        a, b = read(old), read(new)
        # The published design changes the schedule hash inside the card-plan identity.
        # The preparer already proved the original cache key and rebuilt this field.
        if field not in a or dict(a, **{field: b[field]}) != b:
            raise ValueError('previous prepared input differs beyond its schedule binding')


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
    if 'batch_probe' in c:
        from tools.reports.card_batch_comparison import run_probe
        return run_probe(c)
    if 'resume' in c:
        return run_resume(c)
    role = c['role']
    entry, work = Path(c['entry']),Path(c['work'])
    root = work/role
    root.mkdir()
    reuse = c.get('prepared_input')
    init = Path(reuse['root']) if reuse else work/'prepare'
    acoustic = Path(c['prepared_mixed']['root']) if 'prepared_mixed' in c else work/'prepare'
    prior_plan = c.get('fresh_card_plan_sha256') or c.get('cache_card_plan_sha256')
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
            plan_sha = file_hash(entry/'card-plan.json')
            if (view['layout_plan_sha256']!=(prior_plan or plan_sha) or
                view['source_identity']!=training['release_sha256'] or
                view['full_verified']!=training['batch_config']['source_queries'] or
                not (acoustic/'synthetic-cache').is_dir()):
                raise ValueError('mixed layout, release or population changed')
            proof=dict(proof,reused_receipt=c['prepared_mixed']['receipt_sha256'])
            if c.get('cache_card_plan_sha256'):
                commit_json(root/'card-cache.json',dict(view,layout_plan_sha256=plan_sha))
                proof.update(cache_sha256=file_hash(root/'card-cache.json'))
            if c.get('fresh_card_plan_sha256'):
                from tools.reports.initialize_context_stage1_policy import run as initialize
                design = entry/'design.json'
                declared = read(entry/'init.json')
                policy = initialize(dict(declared,design=str(design),design_sha256=file_hash(design)),root)
                if (policy['base_id'] != training['base_id']
                        or policy['processor_sha256'] != view['processor_sha256']
                        or policy['audio_token_id'] != view['encoding']['audio_token_id']):
                    raise ValueError('fresh model does not reproduce cached input identity')
                commit_json(root/'card-cache.json',dict(view,layout_plan_sha256=plan_sha))
                proof.update(policy_id=policy['policy_id'],base_id=policy['base_id'],
                             cache_sha256=file_hash(root/'card-cache.json'))
            if 'single_card' in c:
                from multimodal_jev.single_card_forward import prepare as prepare_single
                reuse_single = c.get('reuse_single')
                if reuse_single:
                    import shutil
                    path = Path(reuse_single['root'])/'single.json'
                    if file_hash(path) != reuse_single['sha256']:
                        raise ValueError('published card token identities differ')
                    spec = read(path)
                    if spec['plan'] != (prior_plan or plan_sha) or spec['config'] != c['single_card']:
                        raise ValueError('published card assignment or template differs')
                    if prior_plan:
                        commit_json(root/'single.json',dict(spec,plan=plan_sha))
                    else:
                        shutil.copyfile(path,root/'single.json')
                    proof['single_card'] = dict(classes=len(spec['labels']),unique_prompts=len(spec['catalog']),questions_per_sequence=1)
                else:
                    proof['single_card']=prepare_single(entry,root/'single.json',init/'policy',c['sample'],c['single_card'])
                proof['single_sha256']=file_hash(root/'single.json')
            if 'published_acoustic_pack' in c:
                published=c['published_acoustic_pack']
                receipt=Path(published['root'])/'receipt.json'
                if file_hash(receipt)!=published['receipt_sha256']:
                    raise ValueError('previous preparation receipt changed')
                prev=read(receipt)
                verify_pack_inputs(prev, proof, receipt.parent, root,
                                   rebind_plan='token_buckets' in c and bool(c.get('cache_card_plan_sha256')))
                pin=prev['packed_acoustics']
                path=Path(published['root'])/'acoustics/index.json'
                if file_hash(path)!=pin['sha256']:raise ValueError('published pack index changed')
                proof['packed_acoustics']=dict(pin,path=str(path))
            elif 'acoustic_pack' in c:
                from multimodal_jev.card_acoustic_pack import build
                proof['packed_acoustics'] = build(root/'acoustics',entry,training,view,
                    acoustic/'synthetic-cache',c['acoustic_pack'])
            commit_json(root/'receipt.json',proof)
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
    from multimodal_jev.card_contrast import CARD_KINDS
    if not finite:
        witness = read(work/'finite/receipt.json')
        objective = c.get('triplet_objective', {})
        if ('token_buckets' in c and witness.get('token_buckets') != c['token_buckets']) or witness['updates'] != c['finite_updates']:
            raise ValueError('two finite updates required')
        if objective.get('kind') in CARD_KINDS and witness.get('instruction_objective') != objective:
            raise ValueError('finite window did not admit this contrastive configuration')
    parent.update(source_root=str(entry/'source'),source_policy=str(init/'policy'),source_policy_id=proof['policy_id'],
        source_contract=manifest['contract'],stage_lora_transition=bind_design(design,manifest['contract']),
        stage1_layout={k:dict(path=str(entry/name),sha256=file_hash(entry/name))
                       for k,name in [('design','design.json'),('plan','card-plan.json')]},
        layout_cache=dict(path=str((work/'prepare' if prior_plan else acoustic if c.get('mixed') else init)/'card-cache.json'),sha256=proof['cache_sha256']),
        stop_update=c['finite_updates'] if finite else budget,engineering_prefix=finite,training_dispatch_allowed=not finite,
        checkpoint_interval=c['finite_updates'] if finite else c['checkpoint_interval'],evaluation_steps=c['evaluation_steps'],
        reference_endpoint=budget,selection_rule='predeclared_final_endpoint',parity_passes=c['parity_passes'] if finite else 0,
        parity_probability_atol=c['parity_probability_atol'])
    if c.get('mixed'):
        parent['mixed'].update(entry=str(entry),cache=str(acoustic/'synthetic-cache'))
    if 'single_card' in c:
        parent['single_card']=dict(path=str(work/'prepare/single.json'),sha256=proof['single_sha256'])
    if 'triplet_objective' in c:
        parent['instruction_objective'] = c['triplet_objective']
        parent['triplet_prefetch'] = c['triplet_prefetch']
        if 'acoustic_pack' in c or 'published_acoustic_pack' in c:
            parent['packed_acoustics'] = proof['packed_acoustics']
        if 'class_logit_ema' in c:
            parent['class_logit_ema'] = c['class_logit_ema']
            parent['class_scoring'] = c['class_scoring']
            parent.pop('epoch_mining', None)
            parent.pop('bootstrap_input', None)
        else:
            parent['epoch_mining'] = c['epoch_mining']
        parent['batch_config']['per_dp_replica_batch'] = c['effective_anchors_per_rank']
        from multimodal_jev.context_topology_batch_plan import derive
        plan = derive(parent['batch_config'])
        budget = plan['total_updates']
        parent.update(global_batch=plan['global_batch'],reference_endpoint=budget,
                      stop_update=c['finite_updates'] if finite else budget,
                      checkpoint_interval=c['finite_updates'] if finite else plan['updates_per_epoch'],
                      evaluation_steps=list(range(0,budget+1,plan['updates_per_epoch'])))
    if not finite and c.get('stop_after_epoch'):
        epoch = c['stop_after_epoch']
        if not 0 < epoch < parent['batch_config']['epochs']:
            raise ValueError('first training window must stop at an interior epoch boundary')
        parent.update(stop_update=plan['updates_per_epoch']*epoch,mine_after_window=False)
    if 'execution_microbatch' in c:
        micro=c['execution_microbatch']
        if type(micro)!=int or micro<1 or parent['batch_config']['per_dp_replica_batch']%micro:
            raise ValueError('microbatch must divide the effective replica batch')
        parent['batch_config']['microbatch']=parent['microbatch']=micro
    widths = c.get('capacity_widths',[parent['microbatch']]) if finite else [
        read(work/'finite/receipt.json').get('microbatch',parent['microbatch'])]
    if 'bootstrap_input' in c:
        reuse = c['bootstrap_input']
        for key in ('config', 'report'):
            if file_hash(reuse[key]['path']) != reuse[key]['sha256']:
                raise ValueError(f'bootstrap {key} changed')
        prior = read(reuse['config']['path'])
        for key in ('source_policy_id', 'source_contract', 'base_id', 'release_sha256',
                    'stage1_layout', 'single_card', 'layout_cache',
                    'mixed', 'epoch_mining', 'runtime_profile'):
            if prior[key] != parent[key]:
                raise ValueError(f'bootstrap identity differs: {key}')
        if prior['physical_devices'] != parent['physical_devices'] and reuse.get('topology_independent') is not True:
            raise ValueError('bootstrap reuse across topologies must be declared')
        parent['bootstrap_input'] = reuse['report']
    if 'token_buckets' in c:
        parent['token_buckets'] = c['token_buckets']
    if 'training_acoustic_engine' in c:
        parent['training_acoustic_engine']=c['training_acoustic_engine']
    if 'training_graph' in c:
        parent['training_graph'] = c['training_graph']
        if not finite:
            receipt_path = work/'finite/receipt.json'
            parent['finite_graph_admission'] = dict(path=str(receipt_path), sha256=file_hash(receipt_path))
    if not finite and 'continuation' in c:
        parent.update(continuation=c['continuation'],start_update=c['continuation']['step'])
    if not finite and 'recovery' in c:
        parent['recovery'] = c['recovery']
    if 'activation_checkpointing' in c:
        parent['activation_checkpointing'] = c['activation_checkpointing']
    if finite and 'finite_pressure' in c:
        parent['finite_pressure'] = c['finite_pressure']
    for width in widths:
        parent['microbatch'] = parent['batch_config']['microbatch'] = width
        commit_json(root/'config.json',parent)
        command = [sys.executable,'-m','torch.distributed.run','--nnodes=1','--node-rank=0',
            '--master-addr='+c['master_addr'],'--master-port='+str(c['master_port']),f"--max-restarts={0 if finite else parent.get('recovery', {}).get('max_restarts', 0)}",
            '--nproc-per-node='+str(len(parent['physical_devices'])),str(entry/'source/tools/jev/train_card_stage1.py'),
            '--config',str(root/'config.json'),'--output',str(root/'run')]
        if not finite or 'capacity_widths' not in c:
            subprocess.run(command,check=True)
            break
        import re
        log = root/f'micro-{width}.log'
        with log.open('x') as output:
            code = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT).returncode
        if code == 0:
            break
        tail = log.read_text(errors='replace')
        print(tail[-5000:],flush=True)
        if not re.search(r'(NPU out of memory|NPUOutOfMemoryError)',tail,re.I) or width == widths[-1]:
            raise RuntimeError(f'finite width {width} failed, see {log}')
        failed = root/f'oom-{width}'
        (root/'run').rename(failed)
        (root/'config.json').rename(failed/'config.json')
    result = read(root/'run/result.json')
    dynamic = parent.get('token_buckets')
    actual_stop = result['stop_update']
    if (result['passed'] is not True or result['engineering_prefix']!=finite
            or ((not dynamic or finite) and actual_stop != parent['stop_update'])
            or type(actual_stop) is not int or actual_stop < 1):
        raise ValueError('stage1 terminal differs')
    max_observed_lag = 0
    pair_total = {}
    for rank in range(len(parent['physical_devices'])):
        rank_root=root/'run'/f'rank-{rank}'
        epoch_counts = []
        epoch_count = 0
        if finite:
            parity = read(rank_root/'inference-parity.json')
            passes = c['parity_passes']
            widths = [parent['microbatch']]
            if 'token_buckets' in parent:
                widths = [read(rank_root/'update-0001.json')['input_service']['local_groups']]
            if parity['inference_passes'] != passes or parity['compared'] != sum(widths[i % len(widths)] for i in range(passes)):
                raise ValueError('finite inference parity incomplete')
        for step in range(parent.get('start_update',0)+1,actual_stop+1):
            record=read(rank_root/f'update-{step:04d}.json')
            if dynamic:
                from bisect import bisect_left
                info = record['input_service']
                label = bisect_left(dynamic['max_tokens'], info['bucket_max_tokens'])
                real = record['real_queries']
                if (info['bucket'] != label or not 0 < info['local_groups'] <= dynamic['max_anchors'][label]
                        or info['local_groups'] != math.ceil(real / len(parent['physical_devices']))
                        or real < 1 or info['epoch'] != len(epoch_counts)):
                    raise ValueError('bucket update geometry or epoch differs')
                epoch_count += real
                if info['epoch_end']:
                    if epoch_count != parent['batch_config']['source_queries']:
                        raise ValueError('incomplete bucket epoch')
                    epoch_counts.append(step)
                    epoch_count = 0
            if 'class_logit_ema' in parent:
                lag = step-1-record['metrics']['sampler_source_update']
                if lag != record['metrics']['sampler_lag'] or not 0 <= lag <= parent['class_logit_ema']['max_staleness_updates']:
                    raise ValueError('sampler source step differs')
                max_observed_lag = max(max_observed_lag, lag)
            if 'speech_partners' in parent:
                from multimodal_jev.speech_pair_readout import tally
                by_type = tally(record['metrics'],record['real_queries'])
                # The counts were all-reduced; each update contributes once, not once per rank.
                if rank == 0:
                    for kind, row in by_type.items():
                        acc = pair_total.setdefault(kind,dict.fromkeys(row,0))
                        for k,n in row.items():
                            acc[k] += n
            if record['update']!=step or not math.isfinite(record['metrics']['actor/pg_loss']):
                raise ValueError('missing optimizer update')
        if dynamic and not finite and (len(epoch_counts) != parent['batch_config']['epochs'] or epoch_count):
            raise ValueError('incomplete bucket training population')
    receipt=dict(microbatch=parent['microbatch'],updates=result['stop_update'],engineering_prefix=finite,config_sha256=file_hash(root/'config.json'))
    if 'training_graph' in parent:
        receipt['training_graph'] = parent['training_graph']
    if dynamic:
        receipt.pop('microbatch')
        receipt.update(token_buckets=dynamic, epoch_end_updates=epoch_counts)
    if 'continuation' in parent:
        receipt.update(start_update=parent['start_update'],stop_update=parent['stop_update'],continuation=parent['continuation'])
    if 'speech_partners' in parent:
        receipt['paired_alignment'] = dict(index_sha256=parent['speech_partners']['sha256'],
                                          before_optimizer_updates=pair_total)
    if parent['instruction_objective']['kind'] in CARD_KINDS:
        receipt['instruction_objective'] = parent['instruction_objective']
    if 'negative_cards_per_anchor' in parent['instruction_objective']:
        receipt['negative_cards_per_anchor'] = parent['instruction_objective']['negative_cards_per_anchor']
    if 'class_logit_ema' in parent:
        receipt['sampler'] = dict(kind='class-logit-ema-v1',config=parent['class_logit_ema'],
            maximum_observed_lag_updates=max_observed_lag,
            expected_updates=actual_stop)
    if 'finite_pressure' in parent:
        receipt['finite_pressure'] = parent['finite_pressure']
    commit_json(root/'receipt.json',receipt)
    return receipt


def run_resume(c):
    """Restore identical optimizer shards in isolated finite and formal runs."""
    from tools.reports.collect_card_first_updates import summarize

    role = c['role']
    finite = role == 'finite'
    if role not in ('finite', 'formal'):
        raise ValueError('unknown resume window')
    entry, work = Path(c['entry']), Path(c['work'])
    root = work/role
    root.mkdir()
    resume = c['resume']
    parent_path = entry/'parent-training.json'
    if file_hash(parent_path) != resume['parent_config_sha256']:
        raise ValueError('parent training config changed')
    parent = read(parent_path)
    start = resume['start_update']
    checkpoint = Path(resume['checkpoint_root'])/f'checkpoint-{start:04d}-committed.json'
    if file_hash(checkpoint) != resume['checkpoint_sha256']:
        raise ValueError('resume checkpoint identity differs')
    stop = start+c['finite_updates'] if finite else parent['reference_endpoint']
    binding = dict(checkpoint_sha256=resume['checkpoint_sha256'], start_update=start)
    if 'class_logit_ema' in parent:
        if read(checkpoint).get('sampler_state') != resume['sampler_state']:
            raise ValueError('committed online sampler differs')
        binding['sampler_state'] = resume['sampler_state']
    else:
        confusion = entry/'confusion.json'
        if (file_hash(confusion) != resume['confusion_sha256']
                or read(confusion)['checkpoint_sha256'] != resume['checkpoint_sha256']):
            raise ValueError('merged confusion identity differs')
        binding['confusion_sha256'] = resume['confusion_sha256']
        parent['resume_confusion'] = dict(path=str(confusion),sha256=resume['confusion_sha256'])
    if not finite:
        proof = read(work/'finite/receipt.json')
        if (any(proof[k] != v for k,v in binding.items()) or proof['engineering_prefix'] is not True
                or proof['stop_update'] != start+c['finite_updates']
                or proof['parity_passes'] != c['parity_passes']):
            raise ValueError('finite resume did not admit this checkpoint and confusion table')
    if finite:
        parent.pop('recovery', None)
    card = dict(parent['single_card'],path=str(Path(resume['prepared_root'])/'single.json'))
    prepared = Path(resume['prepared_root'])
    if Path(parent['source_policy']) == Path('/jev-work/prepare/policy'):
        parent['source_policy'] = str(prepared/'policy')
    if 'layout_cache' in parent and Path(parent['layout_cache']['path']) == Path('/jev-work/prepare/card-cache.json'):
        parent['layout_cache'] = dict(parent['layout_cache'],path=str(prepared/'card-cache.json'))
    parent.update(start_update=start, stop_update=stop, resume_root=str(checkpoint.parent),
                  single_card=card, engineering_prefix=finite, training_dispatch_allowed=not finite,
                  parity_passes=c['parity_passes'] if finite else 0, mine_after_window=False,
                  pins=dict(parent['pins'],**{str(checkpoint):resume['checkpoint_sha256']}))
    commit_json(root/'config.json',parent)
    world = len(parent['physical_devices'])
    subprocess.run([sys.executable,'-m','torch.distributed.run','--nnodes=1','--node-rank=0',
                    '--master-addr='+c['master_addr'],'--master-port='+str(c['master_port']),
                    f"--max-restarts={0 if finite else parent.get('recovery', {}).get('max_restarts', 0)}",'--nproc-per-node='+str(world),
                    str(entry/'source/tools/jev/train_card_stage1.py'),
                    '--config',str(root/'config.json'),'--output',str(root/'run')],check=True)
    result = read(root/'run/result.json')
    if (result['passed'] is not True or result['start_update'] != start
            or result['stop_update'] != stop or result['engineering_prefix'] is not finite):
        raise ValueError('resume terminal differs')
    ranks = [root/'run'/f'rank-{i}' for i in range(world)]
    restored = dict(exact_masters=True, exact_adam=True, completed_updates=start)
    if any(read(rank/'restored.json') != restored for rank in ranks):
        raise ValueError('all-rank optimizer restoration is required')
    if 'class_logit_ema' in parent and any(read(r/'sampler-restored.json') != binding['sampler_state'] for r in ranks):
        raise ValueError('all-rank sampler restoration is required')
    if finite:
        for rank in ranks:
            parity = read(rank/'inference-parity.json')
            if (parity['inference_passes'] != c['parity_passes']
                    or parity['compared'] != c['parity_passes']*parent['microbatch']
                    or not 0 <= parity['max_abs'] <= parent['parity_probability_atol']):
                raise ValueError('resume inference parity incomplete')
        for step in range(start+1,stop+1):
            summarize([read(r/f'update-{step:04d}.json') for r in ranks],step,parent)
    receipt = dict(binding, passed=True, stop_update=stop, engineering_prefix=finite,
                   parity_passes=parent['parity_passes'], ranks=world, config_sha256=file_hash(root/'config.json'))
    commit_json(root/'receipt.json',receipt)
    return receipt
