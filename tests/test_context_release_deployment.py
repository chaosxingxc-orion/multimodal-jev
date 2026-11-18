import pytest
from tools.reports import context_release_deployment as module
from test_prepare_context_cached_training import write


@pytest.mark.parametrize('fault',[None,'missing','writable','shadow','bytes','manifest','publication'])
def test_every_external_release_pin_must_resolve_to_matching_readonly_bytes(tmp_path,fault):
    release=tmp_path/'release';evidence=tmp_path/'evidence';target='/original/reviews/audit.json'
    write(evidence/'reviews/audit.json',{'decision':'admitted'})
    write(release/'publication-config.json',{'pins':{target:module.file_hash(evidence/'reviews/audit.json')}})
    write(release/'MANIFEST.json',{'schema':'slurp-official-clean-intent-v1',
        'files':{'publication-config.json':module.file_hash(release/'publication-config.json')}})
    c=dict(release=str(release),release_sha256=module.file_hash(release/'MANIFEST.json'))
    argv=['docker','create','-v',str(evidence)+':/original:ro']
    if fault=='missing':argv=[]
    elif fault=='writable':argv[-1]=str(evidence)+':/original'
    elif fault=='shadow':argv+=['-v',str(evidence/'reviews')+':/original/reviews']
    elif fault=='bytes':write(evidence/'reviews/audit.json',{'changed':True})
    elif fault=='manifest':c['release_sha256']='wrong'
    elif fault=='publication':write(release/'publication-config.json',{'pins':{}})
    if fault:
        with pytest.raises(ValueError):module.audit(argv,c)
    else:assert module.audit(argv,c)['external_pins']==1


def test_successors_preserve_explicit_release_mount_without_aliasing():
    c=dict(host='host',lease='lease',shared_root='/shared',driver_root='/driver',
        release_evidence={'root':'/evidence','destination':'/original'},tag='not inherited')
    result=module.inherited(c)
    assert result['release_evidence']==c['release_evidence'] and 'tag' not in result
    result['release_evidence']['root']='changed'
    assert c['release_evidence']['root']=='/evidence'


@pytest.mark.parametrize('corrupt',[False,True])
def test_portable_release_checks_bundled_evidence_without_external_mounts(tmp_path,corrupt):
    from test_slurp_official_training import fixture
    from multimodal_jev import slurp_portable_release
    c,_=fixture(tmp_path);root=tmp_path/'portable'
    slurp_portable_release.publish(c,root)
    config=dict(release=str(root),release_sha256=module.file_hash(root/'MANIFEST.json'))
    if corrupt:
        (root/'publication-config.json').write_text('{}')
        with pytest.raises(ValueError):module.audit([],config)
    else:assert module.audit([],config)['external_pins']==0
