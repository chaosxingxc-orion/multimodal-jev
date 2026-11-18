import json
import pytest
from tools import sample_dataset as sample


def fixture(tmp_path):
    path = tmp_path / 'input.jsonl'
    rows = [dict(id=str(i), split='train', label='rare' if i < 2 else 'common', audio=f'{i}.wav') for i in range(10)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    config = dict(schema='training-dataset-sample-v1', input=str(path), input_sha256=sample.sha(path),
        output_directory=str(tmp_path/'out'), seed=317, count=6, id_field='id', split_field='split',
        train_split='train', class_field='label', strategy='balanced')
    return path, rows, config


def test_balanced_without_replacement_and_order_independence(tmp_path):
    path, rows, config = fixture(tmp_path)
    chosen, summary = sample.select(config)
    assert summary['sampled_by_class'] == {'common': 4, 'rare': 2}
    assert len(set(summary['selected_ids'])) == 6
    assert all(json.loads(item[3]) in rows for item in chosen)
    path.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    config['input_sha256'] = sample.sha(path)
    assert sample.select(config)[1] == summary


@pytest.mark.parametrize('defect', ['test', 'duplicate', 'missing_label', 'oversize', 'hash', 'boolean_seed'])
def test_invalid_population_rejected(tmp_path, defect):
    path, rows, config = fixture(tmp_path)
    if defect == 'test': rows[0]['split'] = 'test'
    elif defect == 'duplicate': rows[0]['id'] = rows[1]['id']
    elif defect == 'missing_label': del rows[0]['label']
    elif defect == 'oversize': config['count'] = 11
    elif defect == 'boolean_seed': config['seed'] = True
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    config['input_sha256'] = '0'*64 if defect == 'hash' else sample.sha(path)
    with pytest.raises(ValueError): sample.select(config)


def test_publish_reuse_and_tampering(tmp_path, monkeypatch):
    path, rows, config = fixture(tmp_path)
    template, local = tmp_path/'config.json', tmp_path/'local.json'
    template.write_text(json.dumps(config))
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros={})))
    monkeypatch.setattr(sample, 'check_paths', lambda *a: [])
    assert sample.run(template, local)['status'] == 'complete'
    assert sample.run(template, local)['status'] == 'already_complete'
    (tmp_path/'out/sample.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Source changed'): sample.run(template, local)


def test_workflow_uses_sampler_and_verifies_input(tmp_path, monkeypatch):
    from tools import run_operations as ops
    path, rows, config = fixture(tmp_path)
    template, local = tmp_path/'config.json', tmp_path/'local.json'
    template.write_text(json.dumps(config))
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros={})))
    workflow = tmp_path/'workflow.json'
    workflow.write_text(json.dumps(dict(schema='local-operations-v1', output_directory=str(tmp_path/'workflow'),
        steps=[dict(id='sample', kind='dataset_sample', config=str(template), depends_on=[])])))
    monkeypatch.setattr(sample, 'check_paths', lambda *a: [])
    monkeypatch.setattr(ops, 'check_paths', lambda *a: [])
    assert ops.run(workflow, local)['completed'] == ['sample']
    assert ops.run(workflow, local)['status'] == 'already_complete'
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='Source changed'): ops.run(workflow, local)
