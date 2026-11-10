"""Input-only complete official SFC TEST populations and frozen output contract."""
import hashlib
import json
from pathlib import Path

from .baseline import file_hash
from .consumption import canonical
from .sfc import source_identity
from .sfc_consumption import validate_request

COUNTS = {'test_id': (2590, 1986), 'test_ood': (771, 593)}
RUNTIME = dict(parameters_frozen=True, route='audio-direct', dtype='bfloat16', seed=17,
               max_context_tokens=4096, max_new_tokens=1024, sampling={'temperature': 0., 'n': 1})


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def select_inputs(cfg):
    if (cfg.get('split') not in COUNTS or cfg.get('adapter') is not None
            or any(cfg.get(k) != v for k, v in RUNTIME.items())):
        raise ValueError('registered full official TEST frozen runtime required')
    root = Path(cfg['release']); split = cfg['split']
    manifest_path = root/'MANIFEST.json'; path = root/f'{split}.requests.jsonl'
    if file_hash(manifest_path) != cfg['manifest_sha256'] or file_hash(path) != cfg['requests_sha256']:
        raise ValueError('native release or full requests changed')
    manifest = json.loads(manifest_path.read_text()); pin = manifest['splits'][split]
    if (manifest['identity']['dataset'] != 'sfc-native'
            or set(manifest['splits']) != {'train_id', 'test_id', 'test_ood'}
            or pin['requests_sha256'] != cfg['requests_sha256']
            or (pin['requests'], pin['source_groups']) != COUNTS[split]):
        raise ValueError('complete native official population required')
    rows = []; ids = set(); groups = {}
    with path.open() as source:
        for line in source:
            row = validate_request(json.loads(line))
            if row['id'] in ids or row['split'] != split:
                raise ValueError('unique exact-split requests required')
            # Source IDs, rather than gold metadata, establish conversation
            # coverage. The ordered waveform prefix must agree across stages.
            if not row['id'].startswith('sfc:'):
                raise ValueError('original SFC example identity required')
            relative = row['id'][4:]; identity = source_identity([relative])
            if (identity['official_split'] != split or identity['source_group'] != row['source_group']
                    or identity['stage'] != len(row['input']['audios'])
                    or not row['input']['audios'][-1].endswith('/'+relative)):
                raise ValueError('original conversation identity required')
            stages = groups.setdefault(row['source_group'], {})
            if identity['stage'] in stages:
                raise ValueError('duplicate conversation stage')
            stages[identity['stage']] = row
            rows.append(row); ids.add(row['id'])
    if (len(rows), len(groups)) != COUNTS[split]:
        raise ValueError('full official split required; no subsampling')
    for group, stages in groups.items():
        if set(stages) != ({1, 2} if group.startswith('level_3') else {1}):
            raise ValueError('complete supplied-history conversations required')
        if 2 in stages and (stages[1]['input']['audios'] != stages[2]['input']['audios'][:1]
                            or stages[1]['audio_sha256'] != stages[2]['audio_sha256'][:1]):
            raise ValueError('ordered conversation audio prefix changed')
    return rows


def check_probe(cfg):
    """Bind the already independently accepted TRAIN run, without reading TEST gold.

    The external launcher must also inspect the exact successful predecessor
    container; this local check cannot establish Docker state from inside a job.
    """
    root = Path(cfg['native_probe'])
    if file_hash(root/'acceptance.json') != cfg['native_probe_acceptance_sha256']:
        raise ValueError('registered TRAIN acceptance changed')
    accepted = json.loads((root/'acceptance.json').read_text())
    result = json.loads((root/'run/result.json').read_text())
    identity = json.loads((root/'run/identity.json').read_text())
    prior_cfg = json.loads((root/'config.json').read_text())
    launch = json.loads((root/'launch.json').read_text())
    if (accepted['container_id'] != cfg['native_probe_container_id']
            or launch['container_id'] != accepted['container_id'] or launch['image'] != cfg['image']
            or accepted['requests'] != 6 or accepted['source_groups'] != 4 or accepted['weight_updates'] != 0
            or file_hash(root/'run/result.json') != accepted['result_sha256']
            or file_hash(root/'score/metrics.json') != accepted['score_sha256']
            or json.loads((root/'evaluation-terminal.json').read_text())['exit_code'] != 0
            or json.loads((root/'run/terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or result['model_files_unchanged'] is not True
            or result['weight_updates'] != 0 or result['requests'] != result['predictions'] or result['requests'] != 6
            or file_hash(root/'run/predictions.jsonl') != result['predictions_sha256']
            or identity['config'] != prior_cfg or prior_cfg['split'] != 'train_id'
            or any(prior_cfg.get(k) != v for k, v in RUNTIME.items())
            or any(prior_cfg[k] != cfg[k] for k in ('audio_model', 'versions'))):
        raise ValueError('complete accepted frozen native TRAIN probe required')
    for name, expected in identity['source_files'].items():
        if file_hash(root/'source'/name) != expected:
            raise ValueError('accepted TRAIN source changed')
    for name in ('src/multimodal_jev/sfc_omni.py', 'src/multimodal_jev/sfc_audio.py'):
        if file_hash(Path(__file__).parents[2]/name) != identity['source_files'][name]:
            raise ValueError('full TEST adapter differs from accepted TRAIN')
    return identity


def check_predictions(rows, predictions):
    if len(predictions) != len(rows):
        raise ValueError('complete predictions required')
    for row, p in zip(rows, predictions, strict=True):
        if (p['example_id'] != row['id'] or p['split'] != row['split']
                or p['source_group'] != row['source_group'] or p['request_sha256'] != digest(row)
                or p['audio_sha256'] != row['audio_sha256']
                or p['hf_vllm_prompt_ids_equal'] is not True
                or type(p['input_tokens']) is not int or not 0 < p['input_tokens'] <= 3072
                or type(p['raw_text']) is not str or p['finish_reason'] not in ('stop', 'length')
                or type(p['length_limited']) is not bool or p['length_limited'] != (p['finish_reason'] == 'length')
                or not isinstance(p['token_ids'], list) or len(p['token_ids']) > 1024
                or any(type(t) is not int or t < 0 for t in p['token_ids'])
                or len(p['expected_frames']) != len(row['audio_sha256'])
                or p['expected_frames'] != p['observed_frames']
                or any(type(n) is not int or n <= 0 for n in p['expected_frames'])):
            raise ValueError('full TEST input binding, parity or output budget differs')
        for key in ('prompt_sha256', 'input_ids_sha256'):
            value = p[key]
            if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('valid prompt and token digests required')
