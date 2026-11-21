"""Encode or independently verify the label-free inputs of a native evaluation seal."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.context_decision_contract import (
    ApplicabilityQuestion, ContextDecisionRequest, PREFIX_LAYOUT, render_prefix_context_request,
)
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_training import content_hash


def plan(c):
    root = Path(c['seal']['root'])
    manifest = root/'MANIFEST.json'
    if file_hash(manifest) != c['seal']['sha256']:
        raise ValueError('evaluation seal manifest changed')
    m = json.loads(manifest.read_text())
    if m['schema']=='jev-semantic-evaluation-seal-v1':
        if m['prompt_layout']!=PREFIX_LAYOUT or c.get('panel_size') is not None:
            raise ValueError('full semantic evaluation required')
        jobs=[]
        for name in [c['dataset']] if 'dataset' in c else m['datasets']:
            def bound(file):
                rel=f'inference/{name}/{file}.json'
                if file_hash(root/rel)!=m['files'][rel]:raise ValueError('semantic input changed')
                return json.loads((root/rel).read_bytes())
            panels,rows=bound('panels'),bound('rows')
            if len(rows)!=m['datasets'][name]['recordings'] or len(set(r['id'] for r in rows))!=len(rows):
                raise ValueError('incomplete population')
            for row in rows:
                if set(row)!={'id','audio','audio_sha256','panel'} or row['audio']!='audio/'+row['audio_sha256']+'.wav':
                    raise ValueError('invalid label-free semantic input')
                panel=panels[row['panel']]
                qs=tuple(ApplicabilityQuestion(**q) for q in panel['questions'])
                if len(qs)!=m['datasets'][name]['candidates']:raise ValueError('invalid width')
                asset=dict(id=row['id'],audio=str(root/row['audio']),audio_sha256=row['audio_sha256'])
                jobs.append((name,asset,ContextDecisionRequest(asset['audio'],qs,tuple(panel['requirements']),version=panel['version'])))
        return jobs
    if m['schema'] != 'jev-native-evaluation-seal-v1' or m['prompt_layout'] != PREFIX_LAYOUT:
        raise ValueError('native stable-prefix evaluation seal required')
    datasets = [c['dataset']] if 'dataset' in c else list(m['datasets'])
    size = c.get('panel_size')
    if size is not None and (len(datasets) != 1 or type(size) is not int or size < 1):
        raise ValueError('a positive panel size requires a single sealed dataset')
    jobs = []
    for name in datasets:
        spec = m['datasets'][name]
        files = [root/f'inference/{name}/{file}.json' for file in ('panel', 'rows')]
        for path in files:
            if file_hash(path) != m['files'][str(path.relative_to(root))]:
                raise ValueError('sealed inference input changed: ' + str(path))
        panel, rows = [json.loads(p.read_text()) for p in files]
        questions = tuple(ApplicabilityQuestion(**q) for q in panel['questions'])
        ids = [q.id for q in questions]
        if (panel['history'] or panel['prompt_layout'] != PREFIX_LAYOUT
                or ids != panel['candidate_ids'] or len(ids) != spec['candidates']
                or len(rows) != spec['recordings'] or len({r['id'] for r in rows}) != len(rows)):
            raise ValueError('sealed panel or population differs: ' + name)
        if size is not None:
            chosen = {r['id'] for r in sorted(rows, key=lambda r: (content_hash(r['id']), r['id']))[:size]}
            rows = [r for r in rows if r['id'] in chosen]
        for row in rows:
            if set(row) != {'id', 'audio', 'audio_sha256'}:
                raise ValueError('label-free audio roster required')
            req = ContextDecisionRequest(row['audio'], questions, tuple(panel['requirements']),
                                         version=panel['version'])
            if render_prefix_context_request(req) != (panel['system'], '', panel['readouts']):
                raise ValueError('stable-prefix compiler disagrees with the seal')
            jobs.append((name, row, req))
    return jobs


def run(c, provenance, *, verify=False):
    import torch
    from multimodal_jev.cached_context_preparation import CachedContextPreparer, processor_cache_identity
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
    from multimodal_jev.span_request_identity import tensor_identity
    if list(Path('/dev').glob('davinci[0-9]*')):
        raise ValueError('CPU-only input preparation requires no NPU')
    from multimodal_jev.model_provider import resolve_provider, cache_namespace
    provider = resolve_provider(c)
    namespace = cache_namespace(c)
    torch.set_num_threads(1)
    jobs = plan(c)
    source = Path(c['processor']['policy'])
    identity = processor_cache_identity(source)
    if identity != c['processor']['sha256']:
        raise ValueError('frozen processor identity differs')
    root = Path(c['output'])
    check_paths({'encoded evaluation inputs': root})
    if not verify:
        root.mkdir(parents=True, exist_ok=False)
    receipts = root/'inputs.jsonl'
    manifest = root/'MANIFEST.json'
    if verify:
        m = json.loads(manifest.read_text())
        if (m['inputs_sha256'] != file_hash(receipts)
                or m['recordings'] != len(jobs) or m['seal_sha256'] != c['seal']['sha256']
                or m['processor_sha256'] != identity or m['prompt_layout'] != PREFIX_LAYOUT
                or m.get('model_provider_sha256') != namespace or m['encoding'] != c['encoding']):
            raise ValueError('complete cache manifest for this release required')
        cache = ReadOnlyEncodedAudioCache(root/'cache')

        def factory():
            raise ValueError('missing frozen evaluation input')
    else:
        cache = EncodedAudioCache(root/'cache')

        def factory():
            return provider.processor(source/'processor')

    prepare = CachedContextPreparer(factory, cache, processor_sha256=identity,
                                    prompt_layout=PREFIX_LAYOUT, provider_namespace=namespace, encode=provider.encode, **c['encoding'])

    def encode(job):
        name, row, request = job
        value = prepare((request, row['audio_sha256']))
        key = key_for(prepare.identity(request, row['audio_sha256']))
        ids = value.inputs['input_ids'][0]
        audio = (ids == c['encoding']['audio_token_id']).nonzero().flatten()
        if not len(audio) or int(audio[-1]) >= int(value.positions[0, 0]):
            raise ValueError('audio must precede all decision readouts')
        return dict(dataset=name, id=row['id'], audio_sha256=row['audio_sha256'], cache_key=key,
                    cache_sha256=file_hash(cache.root/(key+'.pt')),
                    tensors={k: tensor_identity(v) for k, v in
                             dict(value.inputs, positions=value.positions).items()})

    with OrderedPrefetch(encode, jobs, workers=c['io_workers'], ahead=c['prefetch']) as work:
        if verify:
            with receipts.open() as stream:
                for new, old in zip(work, stream, strict=True):
                    if new != json.loads(old):
                        raise ValueError('independent verification differs: ' + new['id'])
            return dict(status='full_stable_prefix_cache_verified', recordings=len(jobs),
                        manifest_sha256=file_hash(manifest), model_calls=0, npu_executed=False)
        with receipts.open('x') as stream:
            for index, row in enumerate(work, 1):
                stream.write(json.dumps(row, separators=(',', ':'))+'\n')
                if index % c['progress_interval'] == 0:
                    stream.flush()
                    atomic_json(root/'progress.json', dict(completed=index, total=len(jobs)))
            stream.flush()
            import os
            os.fsync(stream.fileno())
    result = dict(schema='jev-native-seal-encoded-cache-v1', recordings=len(jobs),
                  seal_sha256=c['seal']['sha256'], processor_sha256=identity,
                  prompt_layout=PREFIX_LAYOUT, encoding=c['encoding'], inputs_sha256=file_hash(receipts),
                  model_calls=0, npu_executed=False, provenance=provenance)
    if namespace is not None:
        result['model_provider_sha256'] = namespace
    commit_json(manifest, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    config, meta = load_config(args.config, args.local)
    print(json.dumps(run(config, meta, verify=args.verify), indent=2))
