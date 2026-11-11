import json

import pytest

from multimodal_jev.baseline import file_hash
from multimodal_jev.omni_accepted_sampling import load_accepted_records
from multimodal_jev.omni_sampling_acceptance import audit_records, read_json
from multimodal_jev.sfc_reward_batch import digest
from test_omni_sampling_acceptance import fixture, Tokenizer


def accepted_fixture(tmp_path):
    run, rows, checks, kwargs = fixture(tmp_path)
    write = lambda p, value: p.write_text(json.dumps(value))
    write(tmp_path/'config.json', {'synthetic': True})
    identity = dict(config={'synthetic': True}, inputs_sha256=digest(rows),
        policy_sha256=kwargs['policy_sha256'], instruction_sha256=kwargs['instruction_sha256'],
        effective_sampling=kwargs['sampling'])
    write(run/'identity.json', identity)
    result = read_json(run/'result.json')
    result['identity_sha256'] = file_hash(run/'identity.json')
    write(run/'result.json', result)
    write(tmp_path/'launch.json', dict(container_id='c'*64, source_commit='commit'))
    write(tmp_path/'preparation.json', dict(image='image', source_commit='commit',
        config_sha256=file_hash(tmp_path/'config.json')))
    accepted = dict(audit_records(run, rows, checks, Tokenizer(), **kwargs),
        container_id='c'*64, source_commit='commit', image='image',
        result_sha256=file_hash(run/'result.json'))
    acceptance = tmp_path/'acceptance.json'
    write(acceptance, accepted)
    return dict(root=tmp_path, acceptance=acceptance, acceptance_hash=file_hash(acceptance),
                rows=rows, checks=checks, tokenizer=Tokenizer())


def test_full_raw_records_retained_without_target_files(tmp_path):
    args = accepted_fixture(tmp_path)
    records, identity = load_accepted_records(**args)
    original = [json.loads(line) for line in (tmp_path/'run/records.jsonl').read_text().splitlines()]
    assert records == original and len(records) == 156
    assert records[0]['log_probs'] == [-.3, -.4]
    assert identity['inputs_sha256'] == digest(args['rows'])


@pytest.mark.parametrize('change', ['acceptance', 'source', 'image', 'config', 'rows',
                                  'identity', 'raw', 'records', 'terminal'])
def test_reject_changed_evidence_before_probability_preparation(tmp_path, change):
    args = accepted_fixture(tmp_path)
    if change == 'rows':
        args['rows'][0]['source_group'] = 'changed'
    elif change == 'acceptance':
        args['acceptance_hash'] = '0'*64
    elif change == 'records':
        path = tmp_path/'run/records.jsonl'
        path.write_text(path.read_text().replace('-0.3', '-0.8'))
    else:
        paths = dict(source='launch.json', image='preparation.json', config='config.json',
                     identity='run/identity.json', terminal='run/terminal.json',
                     raw='run/requests/'+digest(('0', '0'))+'.raw.json')
        path = tmp_path/paths[change]
        obj = read_json(path)
        if change == 'source': obj['source_commit'] = 'other'
        if change == 'image': obj['image'] = 'other'
        if change == 'config': obj['synthetic'] = False
        if change == 'identity': obj['policy_sha256'] = 'd'*64
        if change == 'terminal': obj['exit_code'] = 1
        if change == 'raw': obj['outputs'][0]['outputs'][0]['logprobs'][0]['3'] = -.8
        path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        load_accepted_records(**args)
