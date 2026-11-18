import json
from pathlib import Path

import pytest

from tools.reports.prepare_context_dual_throughput_pair import prepare,sha,write
from tools.reports.prepare_context_dual_training_candidate import rebase


def fixture(tmp_path):
    source=tmp_path/'parent';(source/'source').mkdir(parents=True)
    (source/'source/code.py').write_text('pass\n')
    (source/'run.py').write_text('pass\n')
    write(source/'RUNTIME-SOURCES.json',{'code.py':sha(source/'source/code.py')})
    config=dict(start_update=0,stop_update=2,engineering_prefix=True,training_dispatch_allowed=False,
        algorithm='dual_credit_rloo_clipped_v1',dual_arm='joint',checkpoint_interval=1,
        global_batch=128,microbatch=8,reference_cache='/shared/reference',
        pins={'/old/source/code.py':sha(source/'source/code.py')},source_root='/old/source')
    write(source/'config.json',config)
    write(source/'service-config.json',dict(pins=config['pins'],policy=dict(max_batch_queries=4,max_batch_tokens=16384)))
    write(source/'PREPARED.json',dict(status='integrated_prepared_only',remote_root='/old',
        files={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}))
    return dict(source=str(source),source_sha256=sha(source/'PREPARED.json'),output=str(tmp_path/'pair'),remote_root='/new')


def test_pair_changes_only_capacity_and_isolated_paths(tmp_path):
    c=fixture(tmp_path);before=sha(Path(c['source'])/'PREPARED.json');r=prepare(c,{})
    configs=[];services=[]
    for name in ('serial','batch4'):
        p=Path(r['arms'][name]['package']);m=json.loads((p/'PREPARED.json').read_text())
        assert all(sha(p/k)==v for k,v in m['files'].items())
        configs.append(rebase(json.loads((p/'config.json').read_text()),'/new/'+name,'/canonical'))
        services.append(rebase(json.loads((p/'service-config.json').read_text()),'/new/'+name,'/canonical'))
        assert sha(p/'source/code.py')==sha(Path(c['source'])/'source/code.py')
    assert configs[0]==configs[1]
    assert configs[0]['reference_cache']=='/canonical/reference-cache'
    assert services[0]['policy'].pop('max_batch_queries')==1
    assert services[1]['policy'].pop('max_batch_queries')==4
    assert services[0]==services[1]
    assert sha(Path(c['source'])/'PREPARED.json')==before
    assert not r['switch_old_training_allowed']


@pytest.mark.parametrize('fault',['payload','identity','path','existing'])
def test_rejects_changed_or_unsafe_input(tmp_path,fault):
    c=fixture(tmp_path)
    if fault=='payload':(Path(c['source'])/'source/code.py').write_text('changed')
    if fault=='identity':c['source_sha256']='0'*64
    if fault=='path':c['remote_root']='/old/child'
    if fault=='existing':Path(c['output']).mkdir()
    with pytest.raises(ValueError):prepare(c,{})
