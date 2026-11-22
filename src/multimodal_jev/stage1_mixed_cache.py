"""Use the authenticated native acoustic tensors with a separately verified card prefix."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import json

from .baseline import file_hash


def start_worker(root, processor_path, cache_spec):
    import torch
    from transformers import Qwen3ASRProcessor
    from .encoded_audio_cache import EncodedAudioCache
    from .cached_context_preparation import CachedContextPreparer
    from .stage1_card_layout import from_panel
    torch.set_num_threads(1)
    global prepare_one, native_panel
    native_panel = from_panel(cache_spec['templates']['native']['panel'])
    encoding = {k:cache_spec['encoding'][k] for k in ('prompt_layout','max_tokens','decision_token_id','audio_token_id')}
    prepare_one = CachedContextPreparer(partial(Qwen3ASRProcessor.from_pretrained,processor_path,local_files_only=True),
        EncodedAudioCache(root),processor_sha256=cache_spec['processor_sha256'],**encoding)


def encode_row(row):
    sample = prepare_one((replace(native_panel,audio=row['audio']),row['audio_sha256']))
    return row['id'],sample.inputs['input_ids'].shape[-1]


def build(c, entry, root, policy):
    from .context_execution import map_cpu
    from .cached_context_preparation import processor_cache_identity
    from .context_policy_artifacts import read_execution_bundle
    from .local_checkpoint import commit_json
    from .stage1_layout_cache import split_text
    from .stage1_card_layout import from_panel
    from .audio_policy import load_audio
    from .context_decision_contract import encode_context_request
    from transformers import Qwen3ASRProcessor
    def read(p):
        return json.loads(p.read_bytes())
    old = read(policy.parent/'receipt.json')
    if file_hash(policy.parent/'receipt.json') != c['prepared_input']['receipt_sha256']:
        raise ValueError('changed initial policy receipt')
    training = read(entry/'training.json')
    read_execution_bundle(policy,expected_policy_id=old['policy_id'],expected_base_id=training['base_id'],
                          expected_contract=read(policy/'MANIFEST.json')['contract'])
    base = read(entry/'native-cache.json')
    if processor_cache_identity(policy)!=base['processor_sha256']:
        raise ValueError('native processor identity changed')
    plan = read(entry/'card-plan.json')
    rows = read(entry/'synthetic.json')
    spec = c['mixed']
    lengths = read(entry/'native-lengths.json')
    pool = spec['pool']
    # The serial branch of map_cpu runs in the caller and has no initializer.
    args = (str(root/'synthetic-cache'),str(policy/'processor'),base)
    if pool['workers']==1 or len(rows)<pool['min_parallel']:
        start_worker(*args)
    additions,_ = map_cpu(encode_row,rows,initializer=start_worker,initargs=args,**pool)
    lengths.update(additions)
    if set(lengths)!=set(plan['assignments']):
        raise ValueError('missing acoustic lengths')
    sample = c['sample']
    if file_hash(sample['path'])!=sample['sha256']:
        raise ValueError('changed witness audio')
    processor = Qwen3ASRProcessor.from_pretrained(str(policy/'processor'),local_files_only=True)
    audio = load_audio(sample['path'],None)
    enc = {k:base['encoding'][k] for k in ('audio_token_id','decision_token_id','prompt_layout','max_tokens')}
    enc['max_tokens'] = spec['max_tokens']
    def compile_template(panel):
        batch,pos = encode_context_request(processor,from_panel(panel),audio,**enc)
        prefix,suffix,_ = split_text(batch['input_ids'][0],enc['audio_token_id'])
        return dict(panel=panel,prefix=prefix,suffix=suffix)
    original = base['templates']['native']
    if compile_template(original['panel'])!=original:
        raise ValueError('native acoustic layout witness differs')
    templates = {k:compile_template(p) for k,p in plan['panels'].items()}
    deltas = {k:len(t['prefix'])+len(t['suffix'])-len(original['prefix'])-len(original['suffix']) for k,t in templates.items()}
    maximum = max(lengths[uid]+deltas[arm] for uid,pair in plan['assignments'].items() for arm in pair)
    if maximum>enc['max_tokens']:
        raise ValueError('full card input exceeds length limit')
    commit_json(root/'card-cache.json',dict(base,templates=templates,acoustic_template=original,
        acoustic_encoding=base['encoding'],encoding=dict(base['encoding'],max_tokens=enc['max_tokens']),
        schema='jev-card-text-cache-v1',source_identity=training['release_sha256'],full_verified=len(lengths),
        panel_verified=len(templates),max_tokens=maximum,layout_plan_sha256=file_hash(entry/'card-plan.json')))
    commit_json(root/'receipt.json',dict(policy_id=old['policy_id'],population=len(lengths),synthetic=len(rows),
        cache_sha256=file_hash(root/'card-cache.json'),max_tokens=maximum,panels=len(templates),npu_executed=False))


class RoutedCache:
    def __init__(self, real, synthetic, ids):
        self.real,self.synthetic,self.ids=real,synthetic,ids

    def read(self,identity):
        cache = self.synthetic if identity['request']['audio'] in self.ids else self.real
        return cache.read(identity)
