"""Frozen Qwen3-ASR mean-pooled audio + linear classifier baseline.

A closed-label diagnostic control. It does not implement a dynamic Jev head.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def select_examples(manifest, split, per_class):
    from .task_schema import JevExample
    if per_class is not None and (type(per_class) is not int or per_class < 1):
        raise ValueError('per_class must be positive or null for the full official split')
    groups = defaultdict(list)
    with Path(manifest).open() as handle:
        for line in handle:
            ex = JevExample.from_record(json.loads(line))
            if ex.split != split:
                raise ValueError('manifest has unexpected split')
            if ex.metadata.get('variant') == 'native':
                if ex.schema.decision_type != 'choice' or type(ex.target) is not str:
                    raise ValueError('closed-label baseline requires native choice targets')
                groups[ex.target].append(ex)
    if not groups:
        raise ValueError('no native examples')
    selected = []
    for label in sorted(groups):
        examples = sorted(groups[label], key=lambda e: hashlib.sha256(e.example_id.encode()).digest())
        if per_class is not None and len(examples) < per_class:
            raise ValueError(f'{split} class {label} has fewer than {per_class} records')
        selected.extend(examples[:per_class])
    return selected


def prepare_selection(config):
    train = select_examples(config['train_manifest'], 'train', config['train_per_class'])
    valid = select_examples(config['valid_manifest'], 'valid', config['valid_per_class'])
    labels = sorted({ex.target for ex in train})
    if sorted({ex.target for ex in valid}) != labels:
        raise ValueError('train/valid label sets differ')
    for field in ('audio', 'group', 'speaker'):
        def key(ex):
            if field == 'audio': return ex.metadata['audio_sha256']
            if field == 'group': return ex.metadata['source_group']
            return ex.metadata['speaker_id']
        if {key(ex) for ex in train} & {key(ex) for ex in valid}:
            raise ValueError(f'train/valid overlap: {field}')
    return train, valid, labels


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def train_head(config, output):
    import torch
    from .asr import load_runtime, prepare
    torch.set_num_threads(config.get('cpu_threads', 4))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'config': config,
                'train_manifest_sha256': file_hash(config['train_manifest']),
                'valid_manifest_sha256': file_hash(config['valid_manifest']),
                'code_sha256': {name: file_hash(Path(__file__).with_name(name))
                                for name in ('baseline.py', 'asr.py', 'task_schema.py')},
                'weights_sha256': file_hash(Path(config['model_path'])/'model.safetensors')}
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text()) != identity:
            raise ValueError('resume identity mismatch')
    else:
        atomic_json(output/'identity.json', identity)
    if (output/'result.json').exists():
        raise FileExistsError('run already completed')
    train, valid, labels = prepare_selection(config)
    atomic_json(output/'selection.json', {s: [ex.example_id for ex in rows]
                                         for s,rows in [('train',train),('valid',valid)]})
    processor, model = load_runtime(config['model_path'], config['device'], config['dtype'])
    model.requires_grad_(False)
    cache = output/'features'; cache.mkdir(exist_ok=True)
    start = time.monotonic()
    features = []
    for index, ex in enumerate(train+valid):
        name = hashlib.sha256(ex.example_id.encode()).hexdigest()+'.pt'
        path = cache/name
        if path.exists():
            feature = torch.load(path, weights_only=True, map_location='cpu')
        else:
            if file_hash(ex.audio) != ex.metadata['audio_sha256']:
                raise ValueError(f'audio changed: {ex.audio}')
            inputs = prepare(processor, ex.audio, config['device'])
            with torch.inference_mode():
                encoded = model.get_audio_features(
                    inputs['input_features'].to(dtype=model.dtype), inputs['input_features_mask'],
                    return_dict=True).pooler_output
                feature = encoded.mean(dim=0).float().cpu().clone()
            if not torch.isfinite(feature).all():
                raise ValueError('non-finite audio feature')
            temporary = path.with_suffix('.tmp')
            torch.save(feature, temporary); temporary.replace(path)
        features.append(feature)
        atomic_json(output/'progress.json', {'phase':'features','done':index+1,'total':len(train+valid),
                    'heartbeat_unix':time.time(),'elapsed_seconds':time.monotonic()-start})
    # Outside inference mode so the head can save inputs for its backward pass.
    train_x = torch.stack(features[:len(train)]).clone()
    valid_x = torch.stack(features[len(train):]).clone()
    mean, std = train_x.mean(0), train_x.std(0, unbiased=False).clamp_min(1e-6)
    train_x, valid_x = (train_x-mean)/std, (valid_x-mean)/std
    targets = torch.tensor([labels.index(ex.target) for ex in train])
    valid_targets = torch.tensor([labels.index(ex.target) for ex in valid])
    results = []
    for seed in config['seeds']:
        torch.manual_seed(seed)
        head = torch.nn.Linear(train_x.shape[-1],len(labels))
        optimizer = torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'],
                                          foreach=False,fused=False)
        losses=[]
        for step in range(config['steps']):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(head(train_x),targets)
            if not torch.isfinite(loss): raise ValueError('non-finite loss')
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            logits=head(valid_x); predicted=logits.argmax(-1)
            accuracy=float((predicted==valid_targets).float().mean())
            train_accuracy=float((head(train_x).argmax(-1)==targets).float().mean())
        torch.save({'head':head.state_dict(),'mean':mean,'std':std,'labels':labels},output/f'head-{seed}.pt')
        with (output/f'valid-predictions-{seed}.jsonl').open('w') as handle:
            for ex, i, probs in zip(valid,predicted.tolist(),logits.softmax(-1).tolist()):
                handle.write(json.dumps({'example_id':ex.example_id,'value':labels[i],'confidence':probs[i]})+'\n')
        results.append({'seed':seed,'train_accuracy':train_accuracy,'valid_accuracy':accuracy,
                        'first_loss':losses[0],'last_loss':losses[-1]})
    receipt={'status':'passed','route':'frozen-audio-mean-pool-linear-choice',
             'train_count':len(train),'valid_count':len(valid),'classes':len(labels),
             'feature_dim':train_x.shape[-1],'seeds':results,'elapsed_seconds':time.monotonic()-start,
             'scope':config.get('scope','preregistered development pilot, not full paper reproduction or generalization evidence'),
             'identity':identity}
    atomic_json(output/'result.json',receipt)
    return receipt
