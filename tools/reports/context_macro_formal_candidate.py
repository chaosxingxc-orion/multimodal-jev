"""Formal634 candidate over the admitted DP2 trainer; no deployment entry.

Separate from frozen engineering/efficiency packages. Per-update policy export
and dual activation remain mandatory; recovery states use the declared curve.
The formal launcher and independent whole-run auditor are not supplied here.
"""
import ast
from collections import Counter
from pathlib import Path

from tools.reports.context_macro_trainer_candidate import integrated as macro_source, ALGORITHM

STEPS = [0, *range(64, 634, 64), 634]


def contract(config):
    if (config.get('experiment_kind') != 'formal_source'
            or config.get('start_update') != 0 or type(config.get('start_update')) is not int
            or config.get('stop_update') != 634 or type(config.get('stop_update')) is not int
            or config.get('stage') != 'task' or config.get('resume_root')
            or config.get('engineering_prefix') is not False
            or config.get('training_dispatch_allowed') is not True
            or config.get('checkpoint_interval') != 64
            or config.get('evaluation_steps') != STEPS
            or any(type(s) is not int for s in config['evaluation_steps'])
            or config.get('seed') != 317 or type(config.get('seed')) is not int
            or config.get('algorithm') != ALGORITHM or config.get('dual_arm') != 'joint'):
        raise ValueError('fresh formal634 source-only joint GRPO and fixed 11-point curve required')
    return dict(experiment_kind='formal_source', updates=634, seed=317,
        evaluation_steps=list(STEPS), recovery_checkpoint_steps=STEPS[1:],
        behavior_export='every_update_before_dual_replica_ack',
        query_weights='uniform_real_zero_padding', source_selection_only=True)


def validate_schedule(batches, population):
    """Validate actual full two-epoch schedule, including the real denominators."""
    if len(population) != 40513 or len(batches) != 634:
        raise ValueError('complete 40513-recording two-epoch schedule required')
    counts = []
    for epoch in range(2):
        seen = Counter()
        for index, batch in enumerate(batches[epoch*317:(epoch+1)*317]):
            if batch['epoch'] != epoch or batch['index'] != index or len(batch['slots']) != 128:
                raise ValueError('fixed epoch and batch cursor required')
            real = 0
            for slot in batch['slots']:
                if (slot['id'] not in population or type(slot['padding']) is not bool
                        or type(slot['weight']) not in (float, int)
                        or slot['weight'] != float(not slot['padding'])):
                    raise ValueError('known samples with uniform real and zero padding weights required')
                if not slot['padding']:
                    real += 1
                    seen[slot['id']] += 1
            if real != (65 if index == 316 else 128):
                raise ValueError('epoch tail requires 65 real queries and 63 zero-weight slots')
            counts.append(real)
        if seen != Counter(population):
            raise ValueError('each real source sample must appear exactly once per epoch')
    return counts


def generate(source_root):
    # Build on the DP2 integration edits, not the interval transform: the formal
    # curve owns its own recovery-step declaration and block.
    source = macro_source(source_root)
    replacements = [
        ("        if c['engineering_prefix'] is not True or c['training_dispatch_allowed'] is not False:\n"
         "            raise ValueError('continuous candidate is engineering-only, not released')\n"
         "        from tools.reports.context_macro_trainer_candidate import experiment_window\n"
         "        experiment_window(c)",
         "        from tools.reports.context_macro_formal_candidate import contract as formal_contract,validate_schedule\n"
         "        formal=formal_contract(c)"),
        ("        progress('load')", "        contract['formal_experiment']=formal\n        progress('load')"),
        ("                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,",
         "                validate_schedule(batches,population.keys())\n"
         "                session=DualContextRolloutSession(client,cache,reference=reference,initial_policy=initial,"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('formal trainer integration anchor changed: '+before[:90])
        source = source.replace(before, after)
    begin = '                save_now=True\n'
    end = '                activation=[None]\n'
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError('unique export/checkpoint section required')
    left, rest = source.split(begin)
    _, right = rest.split(end)
    source = left + CHECKPOINT_BLOCK + end + right
    ast.parse(source)
    return source


CHECKPOINT_BLOCK = '''                save_now=(index+1) in formal['recovery_checkpoint_steps']
                if frozen!={n:tensor_identity(p.detach().cpu()) for n,p in wrapped.named_parameters() if not p.requires_grad}:
                    raise ValueError('frozen backbone changed')
                if save_now:
                    state=capture(named,opt,rank=rank,world_size=world,completed_updates=index+1,contract=contract)
                    commit_file(out/f'checkpoint-{index+1:04d}.pt',lambda f:torch.save(state,f))
                # Every optimizer update exports only trainable LoRA/head tensors.
                with FSDP.summon_full_params(wrapped,writeback=False,rank0_only=False):
                    full=canonical_execution({n:p.detach().cpu().to(torch.bfloat16).clone()
                        for n,p in model.named_parameters() if p.requires_grad})
                    hashes={n:tensor_identity(v) for n,v in full.items()};peers=[None]*world
                    dist.all_gather_object(peers,hashes)
                    if any(v!=hashes for v in peers):raise ValueError('assembled execution differs across ranks')
                    if rank==0:
                        commit_file(root/f'execution-{index+1:04d}.pt',lambda f:torch.save(full,f))
                        joint=export_checkpoint_policy(c['source_policy'],root/f'policy-{index+1:04d}',full,
                            policy_id=c['source_policy_id'],contract=c['source_contract'],base_id=c['base_id'])
                dist.barrier()
                if rank==0:
                    published=dict(updates=index+1,contract=contract,
                        execution_file=f'execution-{index+1:04d}.pt',
                        joint_policy=joint,joint_policy_directory=f'policy-{index+1:04d}',
                        joint_policy_manifest_sha256=file_hash(root/f'policy-{index+1:04d}/MANIFEST.json'),
                        execution_sha256=file_hash(root/f'execution-{index+1:04d}.pt'))
                    commit_json(root/f'policy-{index+1:04d}-committed.json',published)
                    if save_now:
                        commit_json(root/f'checkpoint-{index+1:04d}-committed.json',dict(published,
                            rank_shards={f'rank-{r}/checkpoint-{index+1:04d}.pt':
                                file_hash(root/f'rank-{r}/checkpoint-{index+1:04d}.pt') for r in range(world)}))
                dist.barrier()
'''


def run(config, output):
    contract(config)
    namespace = {'__name__': __name__+'.derived'}
    exec(compile(generate(config['source_root']), '<formal-macro-dp2-trainer>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(args.config.read_text()),args.output)
