"""Declared-DP GRPO candidate consuming sealed Stage1 results and inputs.

The inherited dual-credit implementation currently supports one fixed objective.
Require its explicit configuration instead of silently ignoring requested tuning.
"""
import ast
from pathlib import Path
from types import FunctionType

from tools.reports.context_replica_trainer_candidate import generate as finite_source
from tools.reports.context_replica_formal_candidate import generate as formal_source
from tools.reports.context_macro_trainer_candidate import integrated


BASELINE_OBJECTIVE = dict(
    algorithm='dual_credit_grpo_floor_clipped_v1', arm='joint', group_size=16,
    clip_ratio=.2, advantage_std_floor=.25, ranking_rl_weight=.5,
    existence_rl_weight=.5, ranking_auxiliary_coefficient=.1,
    binary_nll_coefficient=.3, reference_bernoulli_kl_coefficient=.01,
    learning_rate=1e-4,
)


def validate_objective(config):
    objective = config.get('stage2_objective')
    if not isinstance(objective, dict) or set(objective) != set(BASELINE_OBJECTIVE):
        raise ValueError('explicit complete stage2_objective required')
    for key, expected in BASELINE_OBJECTIVE.items():
        value = objective[key]
        numeric = type(expected) in (int, float)
        if ((numeric and type(value) not in (int, float))
                or (not numeric and type(value) is not str) or value != expected
                or (key == 'group_size' and type(value) is not int)):
            raise ValueError('unsupported Stage2 objective: '+key)
        # Reject contradictory legacy/top-level knobs, too.
        if key in config and (type(config[key]) is not type(value) or config[key] != value):
            raise ValueError('conflicting Stage2 objective: '+key)
    if config.get('algorithm') != objective['algorithm'] or config.get('dual_arm') != objective['arm']:
        raise ValueError('Stage2 algorithm/arm differ from objective')
    return dict(objective)


def no_online_processor():
    raise ValueError('prebuilt input cache required; online preprocessing forbidden')


def generate(source_root, *, formal=False):
    if formal:
        # Formal owns its sparse checkpoint block. The finite generator already
        # replaces that block with interval-based saving, losing the formal
        # anchor. Compose formal on the shared integration BEFORE that edit.
        replica=FunctionType(finite_source.__code__,
            dict(finite_source.__globals__,macro_source=integrated),finite_source.__name__)
        generator=FunctionType(formal_source.__code__,
            dict(formal_source.__globals__,replica_source=replica),formal_source.__name__)
        source=generator(source_root)
    else:
        source=finite_source(source_root)
    edits=[
        ("max_tokens=4096,audio_mirror=audio_mirror)",
         "max_tokens=4096,audio_mirror=audio_mirror,prompt_layout=c.get('prompt_layout'))"),
        ('def run(c,root):\n',
         'def run(c,root):\n'
         '    from tools.reports.context_cached_reference_trainer import validate_objective,no_online_processor\n'
         '    objective_contract=validate_objective(c)\n'),
        ('from multimodal_jev.encoded_audio_cache import EncodedAudioCache',
         'from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache as EncodedAudioCache'),
        ("                require_training_storage(c['encoded_audio_cache'])",
         '                # Input cache is read-only; runtime output storage was checked above.'),
        ("                factory=lambda:Qwen3ASRProcessor.from_pretrained(bundle['processor'],local_files_only=True)",
         '                factory=no_online_processor'),
        ("                cache=RuntimeReferenceCache(c['reference_cache'],policy_id=c['source_policy_id'],\n"
         "                    base_id=c['base_id'],contract=c['source_contract'],runtime_identity=runtime_identity)",
         "                from tools.reports.context_cached_reference_session import load as load_reference\n"
         "                if set(c['reference_dataset']['population'])!=set(population):\n"
         "                    raise ValueError('complete TRAIN reference coverage required')\n"
         "                cache=load_reference(c['reference_dataset'],runtime_identity=runtime_identity,\n"
         "                    policy_id=c['source_policy_id'],base_id=c['base_id'],contract=c['source_contract'],\n"
         "                    candidate_ids=[v['id'] for v in task['schema']['candidates']])"),
        ("        progress('load')",
         "        from multimodal_jev.context_stage_lora import admit_stage2\n"
         "        contract['stage_lora_transition']=admit_stage2(c['stage_lora_transition'],c['source_contract'])\n"
         "        contract['reference_dataset']=dict(manifest_sha256=c['reference_dataset']['manifest_sha256'],\n"
         "            identity=c['reference_dataset']['identity'],mode='readonly_no_online_fill')\n"
         "        contract['stage2_objective']=objective_contract\n"
         "        progress('load')"),
    ]
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('reference dataset integration anchor changed')
        source=source.replace(before,after)
    ast.parse(source)
    return source


def run(config,output):
    namespace={'__name__':__name__+'.derived'}
    source=generate(config['source_root'],formal=config['experiment_kind']=='formal_source')
    exec(compile(source,'<cached-reference-trainer>','exec'),namespace)
    return namespace['run'](config,Path(output))


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run(json.loads(args.config.read_text()),args.output)
