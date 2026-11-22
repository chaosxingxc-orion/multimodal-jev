"""Freeze all clean synthetic recordings with the configured annotation errata."""
from collections import Counter
import json
from pathlib import Path
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import map_io
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.slurp_official_training import inspect_audio


def checked(path, digest):
    if file_hash(path) != digest:
        raise ValueError(f'changed source: {path}')
    return Path(path)


def audit_audio(row):
    row['audio_sha256'] = file_hash(row['audio'])
    row['waveform_sha256'] = inspect_audio(row)['waveform_sha256']
    return row


def admit(c):
    root = checked(c['manifest'], c['pins']['real']).parent
    real_manifest = json.loads((root/'MANIFEST.json').read_bytes())
    tasks = checked(root/'tasks.json', real_manifest['files']['tasks.json'])
    pairs = json.loads(tasks.read_bytes())['intent']['candidate_pairs']
    labels = {tuple(pair): label for label, pair in pairs.items()}
    for label, card in c['extra_cards'].items():
        pair = tuple(card['pair'])
        if pair in labels or label in labels.values():
            raise ValueError('duplicate label extension')
        labels[pair] = label
    errors = {e['slurp_id']:e for e in c['annotation_errors']}
    seen = set()
    admitted, excluded = [], []
    source = checked(c['synthetic'], c['pins']['synthetic'])
    with source.open() as stream:
        for line in stream:
            a = json.loads(line)
            uid, pair = a['slurp_id'], (a['scenario'], a['action'])
            if uid in errors:
                e = errors[uid]
                if (uid in seen or e['status']!='annotation_error' or list(pair)!=e['pair']
                    or a['intent']!=e['intent'] or len(a['recordings'])!=e['recordings']):
                    raise ValueError('configured erratum differs from the frozen annotation')
                seen.add(uid)
                excluded.extend(dict(slurp_id=uid,id='slurp:'+r['file'],status=e['status'],
                                     scenario=a['scenario'],action=a['action'],intent=a['intent']) for r in a['recordings'])
                continue
            label = labels[pair]
            for r in a['recordings']:
                name = r['file']
                if Path(name).name!=name or r['status']!='synthetic':
                    raise ValueError('unexpected synthetic recording identity')
                admitted.append(dict(id='slurp:'+name,audio=str(Path(c['audio'])/name),
                    intent=label,source_group=f'slurp-source:{uid}',role='learning',official_split='train',
                    known_pending_source=False,annotation_text=a['sentence'],slurp_id=uid))
    ids = [r['id'] for r in admitted+excluded]
    if seen!=set(errors) or len(set(ids))!=len(ids):
        raise ValueError('duplicate recording identity or missing erratum')
    admitted, _ = map_io(audit_audio, admitted, workers=c['io_workers'])
    target = Path(c['output'])
    target.mkdir(parents=True,exist_ok=True)
    commit_json(target/'annotation-errors.json',excluded)
    commit_json(target/'synthetic.json',admitted)
    real_count = real_manifest['counts']['learning']['recordings']
    commit_json(target/'admission.json',dict(schema='jev-stage1-acoustic-coverage-v1',
        real=real_count,synthetic=len(admitted),population=real_count+len(admitted),
        clean_synthetic_candidates=len(ids),annotation_errors=len(excluded),
        errors_sha256=file_hash(target/'annotation-errors.json'),source_pins=c['pins'],
        roster_sha256=file_hash(target/'synthetic.json'),synth_classes=Counter(r['intent'] for r in admitted)))


if __name__=='__main__':
    admit(load_config(*sys.argv[1:])[0])
