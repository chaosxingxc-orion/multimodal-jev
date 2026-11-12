"""Compare original D and explicit-correctness engineering replay factors."""
from .intent_objective import identity
from .intent_optimizer import optimizer_kwargs


def validate_factors(reference,replay):
    wanted=identity({'arm':'D'})
    if any(identity(c)!=wanted or c.get('arm')!='D' or c.get('step')!=1 for c in (reference,replay)):
        raise ValueError('second-update D correctness replay required')
    if optimizer_kwargs(reference)!=optimizer_kwargs(replay):raise ValueError('optimizer factors differ')
    relocated={'pins','previous_root','previous_suite','initial_adapter','acceptance_root','objective','adam_betas'}
    # Compare every other field, including future fields not anticipated here.
    if {k:v for k,v in reference.items() if k not in relocated}!={k:v for k,v in replay.items() if k not in relocated}:
        raise ValueError('non-path training factors differ')
    return dict(passed=True,objective=wanted,scope='Configuration factors only; actual policy/state comparison also required.')


def validate_rank(reference,replay,rank):
    for key in ('rank','world_size','updates','arm','base_id','initial_policy_id','adapter_id','recording_ids',
                'admission','execution_dtypes','frozen_base_unchanged','cross_rank_export_exact','actor_sha256'):
        if reference[key]!=replay[key]:raise ValueError('rank replay differs: '+key)
    if replay['rank']!=rank or replay['updates']!=2 or replay['world_size']!=4:
        raise ValueError('exact DP4 rank and second update required')
