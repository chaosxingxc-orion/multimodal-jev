"""Read-only, model-independent Stage1 acoustic inputs. No cards, draws or labels are cached."""
import hashlib
import json
from pathlib import Path
import time
from threading import Lock

import torch

from .baseline import file_hash
from .context_execution import map_cpu
from .encoded_audio_cache import EncodedAudioCache
from .local_checkpoint import commit_file, commit_json
from .slurp_training import content_hash

SCHEMA = 'jev-card-acoustic-pack-v1'


def encode(inputs, positions, prefix, suffix, token, marker):
    ids = inputs['input_ids'][0]
    n = len(ids)-len(prefix)-len(suffix)
    if (set(inputs) != {'input_ids','attention_mask','input_features','input_features_mask'}
            or not len(prefix) or not len(suffix) or n < 1 or
            not torch.equal(ids[:len(prefix)],prefix) or not torch.equal(ids[-len(suffix):],suffix)
            or not (ids[len(prefix):len(prefix)+n] == token).all()
            or (ids == token).sum().item() != n or not (inputs['attention_mask']==1).all()
            or not torch.equal(positions,(ids==marker).nonzero().T)):
        raise ValueError('cached audio expansion differs from authenticated template')
    features, mask = inputs['input_features'], inputs['input_features_mask']
    if (features.dtype != torch.bfloat16 or features.ndim != 3 or features.shape[0]!=1
            or mask.shape != (1, features.shape[-1]) or mask.dtype not in (torch.int64,torch.int32,torch.bool)):
        raise ValueError('unsupported native feature or mask geometry')
    raw = b''.join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in (features,mask))
    return raw, [n, *features.shape[1:], str(mask.dtype).split('.')[-1], hashlib.sha256(raw).hexdigest()]


def _init(root, real, synth, common, prefix, suffix, token, marker):
    global _state
    torch.set_num_threads(1)
    _state = (Path(root), [EncodedAudioCache(real),EncodedAudioCache(synth)], common,
              torch.tensor(prefix), torch.tensor(suffix), token, marker)


def _write(job):
    part, rows = job
    root, caches, common, prefix, suffix, token, marker = _state
    entries = {}
    def write(f):
        for uid, audio, sha, synth in rows:
            if file_hash(audio) != sha:
                raise ValueError('source recording changed: '+uid)
            identity=dict(common,audio_sha256=sha,request=dict(common['request'],audio=audio))
            item = caches[synth].read(identity)
            if item is None:
                raise ValueError('missing authenticated acoustic tensor: '+uid)
            raw, geom = encode(*item[:2],prefix,suffix,token,marker)
            entries[uid] = [part,f.tell(),len(raw),sha,*geom]
            f.write(raw)
    commit_file(root/f'part-{part}.bin',write)
    commit_json(root/f'part-{part}.json',entries)
    return len(entries)


def build(root, entry, train, spec, synth_cache, options):
    start = time.monotonic()
    release = Path(train['release'])
    def pinned(path, digest):
        if file_hash(path)!=digest:
            raise ValueError('pack input changed: '+str(path))
        return path
    real_proof=json.loads(pinned(release/'MANIFEST.json', train['mixed']['real_release_sha256']).read_bytes())
    real=[json.loads(line) for line in pinned(release/'learning.jsonl', real_proof['files']['learning.jsonl']).read_text().splitlines()]
    admission=json.loads(pinned(entry/'admission.json',train['release_sha256']).read_bytes())
    synth=json.loads(pinned(entry/'synthetic.json',admission['roster_sha256']).read_bytes())
    if any(r['role']!='learning' or r['official_split']!='train' for r in real+synth):
        raise ValueError('pack may only contain source TRAIN')
    rows=sorted((r['id'],r['audio'],r['audio_sha256'],i) for i,items in enumerate((real,synth)) for r in items)
    n=len(rows)
    template=spec['acoustic_template']
    enc=spec['acoustic_encoding']
    common=dict(schema='encoded-audio-cache-v1',request=template['panel'],preprocessing=None,
                processor_sha256=spec['processor_sha256'],context_encoding=enc)
    if len({r[0] for r in rows})!=n or n!=spec['full_verified'] or n!=train['batch_config']['source_queries']:
        raise ValueError('incomplete acoustic population')
    size=options['shard_size']
    if type(size) is not int or size<1:
        raise ValueError('positive pack shard size required')
    root.mkdir()
    jobs=[(i//size,rows[i:i+size]) for i in range(0,n,size)]
    kwargs=dict(workers=options['cpu_workers'],window=options['cpu_workers'],min_parallel=1)
    args=(str(root), train['encoded_audio_cache'], str(synth_cache),common,template['prefix'],template['suffix'],
          enc['audio_token_id'], enc['decision_token_id'])
    if kwargs['workers']==1 or len(jobs)==1:
        _init(*args)
    counts,_=map_cpu(_write,jobs,initializer=_init,initargs=args,**kwargs)
    entries={}
    for part,_ in jobs:
        entries.update(json.loads((root/f'part-{part}.json').read_bytes()))
    if sum(counts)!=n or len(entries)!=n:
        raise ValueError('pack publication incomplete')
    key=content_hash(dict(common=common,template=template,release=train['release_sha256']))
    index=dict(schema=SCHEMA,identity=key,token=enc['audio_token_id'],entries=entries)
    commit_json(root/'index.json',index)
    return dict(path=str(root/'index.json'),sha256=file_hash(root/'index.json'), identity=key,
                population=n,build_seconds=time.monotonic()-start,parts=len(jobs))


class PackedAcoustics:
    def __init__(self, pin, population, token, *, allow_subset=False):
        self.path=Path(pin['path'])
        if file_hash(self.path)!=pin['sha256']:
            raise ValueError('acoustic pack index changed')
        index=json.loads(self.path.read_bytes())
        if (index['schema']!=SCHEMA or index['token']!=token or index['identity']!=pin['identity']
                or not (population.keys() <= index['entries'].keys() if allow_subset
                        else population.keys() == index['entries'].keys())
                or any(row['audio_sha256']!=index['entries'][uid][3] for uid,row in population.items())):
            raise ValueError('acoustic pack and TRAIN population differ')
        self.entries=index['entries']
        self.token=token
        self.parts={}
        self.lock=Lock()

    def read(self, uid):
        part, offset, size, _, n, mel, frames, dtype, sha=self.entries[uid]
        if part not in self.parts:
            with self.lock:
                if part not in self.parts:
                    path=self.path.parent/f'part-{part}.bin'
                    self.parts[part]=torch.from_file(str(path),shared=False,size=path.stat().st_size,dtype=torch.uint8)
        raw=self.parts[part].narrow(0,offset,size)
        # Authenticate the exact mapped bytes used for training, including original feature padding.
        if hashlib.sha256(memoryview(raw.numpy())).hexdigest()!=sha:
            raise ValueError('packed tensor bytes changed: '+uid)
        split=2*mel*frames
        def tensor(section,dt):
            # handle unaligned records even when the native mask dtype is bool
            if section.storage_offset()%torch.empty((),dtype=dt).element_size():
                section=section.clone()
            return section.view(dt)
        return dict(input_ids=torch.full((1,n),self.token,dtype=torch.long),
                    input_features=tensor(raw[:split],torch.bfloat16).reshape(1,mel,frames),
                    input_features_mask=tensor(raw[split:],getattr(torch,dtype)).reshape(1,frames))
