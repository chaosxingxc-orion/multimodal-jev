"""Construct an isolated, identity-bound variant of the pinned native actor.

Original files and module globals are untouched. The generated module owns its
own loss registry name and ContextVar. This is not a training launcher.
"""
import hashlib
import json
from pathlib import Path
import sys
import types
from tools.reports.context_attribute_reward_candidate import layout

ACTOR_SOURCE_SHA = '067297bc726d2b607776d40cf6e155f98649ba9405f1d025b43d803e9c1215b0'
DESIGN_SHA = '69c81b2840c295c9504060aa2e4f8b66f0fe29facbf50d0672840db1994f01bd'
CATALOG_SHA = '67ab6400a127a7e06c55e40b0e20a965978bb04f830df11efd383b7bc4813f1f'
REFERENCE_ID = '9164eddffd9e3fdb21c060a72827b1a4324f3d45e60eb0d491b589b194db0e66'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def generate(menu, arm, *, source_root):
    """Return exact derived source + contract for packaging/independent review."""
    root = Path(source_root)
    source = root/'src/multimodal_jev/context_native_actor.py'
    design = root/'configs/experiments/context-native-attribute-comparison-design-v1.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != ACTOR_SOURCE_SHA:
        raise ValueError('Pinned native actor source changed')
    if hashlib.sha256(design.read_bytes()).hexdigest() != DESIGN_SHA:
        raise ValueError('Prospective experiment design changed')
    if digest(menu) != CATALOG_SHA:
        raise ValueError('Frozen native catalog changed')
    nodes, _ = layout(menu, arm)
    contract = dict(schema='context-attribute-actor-binding-v1', arm=arm,
        catalog_sha256=CATALOG_SHA, design_sha256=DESIGN_SHA,
        native_actor_sha256=ACTOR_SOURCE_SHA, reference_policy_id=REFERENCE_ID,
        decision_ids=[n['id'] for n in nodes], group_size=16,
        arithmetic='context-bernoulli-probability-cpu-log-v2')
    identity = digest(contract)
    # A unique registry/ContextVar namespace prevents collisions with live or
    # other candidate actors even when all are imported in the same process.
    replacements = [
        ('from .context_binary_policy import binary_logprobs',
         'from tools.reports.context_probability_policy_v2 import binary_logprobs'),
        ('from .context_hybrid_objective import loss',
         'from tools.reports.context_attribute_objective_candidate import loss as attribute_loss'),
        ('from .context_set_reward import set_f1',
         'from tools.reports.context_attribute_reward_candidate import rewards as attribute_rewards'),
        ("MODE='jev_context_hybrid_v1'", "MODE='jev_attribute_"+identity+"'"),
        ("ContextVar('jev_context_hybrid_active'", "ContextVar('jev_attribute_"+identity+"'"),
        ('algorithm=\'grpo_v1\', kl_coefficient=0., reference_policy_id=None):',
         'algorithm=\'grpo_v1\', kl_coefficient=0., reference_policy_id=None, attribute_contract_id=None):'),
        ('    from verl import DataProto\n',
         "    from verl import DataProto\n"
         "    if (attribute_contract_id != ATTRIBUTE_ID or stage != 'task' or kl_coefficient != .01\n"
         "            or reference_policy_id != ATTRIBUTE_CONTRACT['reference_policy_id']):\n"
         "        raise ValueError('Attribute task contract or reference differs')\n"),
        ("        p={k:item[k] for k in ('request_id','policy_id','inputs','positions','labels','observed',",
         "        if (item.get('attribute_contract_id') != ATTRIBUTE_ID\n"
         "                or item.get('decision_ids') != ATTRIBUTE_CONTRACT['decision_ids']):\n"
         "            raise ValueError('Rollout attribute binding or readout order differs')\n"
         "        p={k:item[k] for k in ('attribute_contract_id','decision_ids','request_id','policy_id','inputs','positions','labels','observed',"),
        ('            rewards=set_f1(actions,labels,observed)',
         "            if actions.shape[1] != 16:raise ValueError('Frozen G16 required')\n"
         "            rewards,_=attribute_rewards(actions,labels,observed,ATTRIBUTE_MENU,arm=ATTRIBUTE_CONTRACT['arm'])"),
        ("                                      'kl_coefficient','reference_policy_id')}",
         "                                      'kl_coefficient','reference_policy_id','attribute_contract_id')}"),
    ]
    text = source.read_text()
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Native actor transformation anchor changed: '+old)
        text = text.replace(old, new)
    header = ("ATTRIBUTE_MENU="+repr(json.loads(json.dumps(menu)))+"\n"
              "ATTRIBUTE_CONTRACT="+repr(contract)+"\nATTRIBUTE_ID="+repr(identity)+"\n")
    wrapper = '''
def loss(logits, labels, observed, request_weights, real_requests, *, stage,
         global_real_requests, world_size, ranking_coefficient, probability_coefficient,
         actions, old_joint, advantages, reference_policy, kl_coefficient, clip_ratio):
    if stage != 'task' or kl_coefficient != .01 or clip_ratio != .2:
        raise ValueError('Frozen task objective required')
    return attribute_loss(logits,labels,observed,request_weights,real_requests,ATTRIBUTE_MENU,
        arm=ATTRIBUTE_CONTRACT['arm'],global_real_requests=global_real_requests,world_size=world_size,
        ranking_coefficient=ranking_coefficient,probability_coefficient=probability_coefficient,
        actions=actions,old_joint=old_joint,advantages=advantages,reference_policy=reference_policy)
'''
    text = header + text + wrapper
    compile(text, '<context-attribute-actor-'+identity+'>', 'exec')
    return text, dict(contract, contract_id=identity,
                      generated_source_sha256=hashlib.sha256(text.encode()).hexdigest())


def load(menu, arm, *, source_root):
    text, receipt = generate(menu, arm, source_root=source_root)
    name = 'multimodal_jev._attribute_candidate_'+receipt['contract_id']
    if name in sys.modules:
        module = sys.modules[name]
        if (module.GENERATED_SOURCE_SHA256 != receipt['generated_source_sha256']
                or digest(module.ATTRIBUTE_MENU) != CATALOG_SHA
                or digest(module.ATTRIBUTE_CONTRACT) != receipt['contract_id']):
            raise ValueError('Loaded candidate identity changed')
        return module, receipt
    module = types.ModuleType(name)
    module.__package__ = 'multimodal_jev'
    module.GENERATED_SOURCE_SHA256 = receipt['generated_source_sha256']
    exec(compile(text, '<'+name+'>', 'exec'), module.__dict__)
    sys.modules[name] = module
    return module, receipt
