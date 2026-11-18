"""Audit failure controls using real prepared inputs, synthetic model outputs."""
import copy
import json
from pathlib import Path

import pytest

from tools.reports.audit_context_attribute_parity_npu import audit, bind_inputs
from tools.reports.launch_context_attribute_parity_npu_v2 import validate
from tools.reports.probe_context_attribute_parity_v2 import content_hash, compare
from tools.reports.build_context_attribute_candidate import request, digest
from tools.reports.prepare_context_attribute_parity_npu_v2 import sha

BASE = Path('.local/decision-objective-research')
PREPARED = BASE/'context-attribute-parity-npu-v2-prepared'
CPU = BASE/'context-attribute-parity-prepare-cpu-v4-collected'
PROOF = Path('docs/research/context-attribute-parity-prepare-fixed-image-cpu-v4.json')


@pytest.fixture(scope='module')
def inputs():
    receipt, config = validate(PREPARED)
    menu = json.loads((PREPARED/'package/input-fixture/catalog.json').read_text())
    anchors = json.loads((CPU/'run/prepared.json').read_text())
    panel = []
    for arm in ('direct','q_input'):
        for asset in config['audio']:
            req = request(asset['audio'],menu,arm=arm)
            panel.append(dict(id=asset['id'],arm=arm,request_sha256=digest(req.model_input()),
                              decision_ids=[q.id for q in req.questions]))
    records = []
    for entry, anchor in zip(panel,anchors['inputs'],strict=True):
        identity = dict(policy_id=config['policy_id'],positions=copy.deepcopy(anchor['positions'][0]),
                        inputs=copy.deepcopy(anchor['tensors']))
        records.append(dict(entry,policy_id=config['policy_id'],request_identity=identity,
                            request_id=content_hash(identity),prob=[.25]*len(entry['decision_ids'])))
    hf = dict(policy_id=config['policy_id'],backend='hf',input_panel=panel,records=records)
    vl = copy.deepcopy(hf);vl['backend']='vllm'
    return receipt,config,menu,anchors,hf,vl


def fixture(tmp_path, inputs):
    receipt, config, menu, anchors, hf, vl = copy.deepcopy(inputs)
    root = tmp_path/'collection';root.mkdir();(root/'evaluation').mkdir()
    def write(name,value):
        (root/name).write_text(json.dumps(value,indent=2)+'\n')
    (root/'config.json').write_bytes((PREPARED/'package/probe-config.json').read_bytes())
    write('evaluation/hf.json',hf);write('evaluation/vllm.json',vl)
    write('evaluation/parity.json',compare(hf,vl))
    for name in ('hf','vllm'):write('evaluation/'+name+'-runtime.json',dict(profile=config['contract']['runtime_profile']))
    write('evaluation/terminal.json',dict(exit_code=0,source_manifest_sha256=receipt['source_manifest_sha256']))
    launch = dict(container_id='synthetic-container',image=receipt['image'],argv=receipt['argv'],
                  source_manifest_sha256=receipt['source_manifest_sha256'],remote_root=receipt['remote_root'],
                  config_sha256=sha(root/'config.json'))
    argv = receipt['argv']
    def vals(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    state = dict(Status='exited',ExitCode=0,OOMKilled=False)
    observed = dict(Id=launch['container_id'],Image=receipt['image'],State=state,
        Config=dict(Entrypoint=vals('--entrypoint'),Cmd=argv[argv.index(receipt['image'])+1:],Env=vals('-e')),
        HostConfig=dict(Binds=vals('-v'),NetworkMode='none',Devices=[
            dict(PathOnHost=x.split(':')[0],PathInContainer=x.split(':')[1],CgroupPermissions='rwm')
            for x in vals('--device')]))
    transport = dict(status='transport_complete_independent_audit_required',container_id=launch['container_id'],
                     image=receipt['image'],remote_root=receipt['remote_root'],state=state,
                     files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()})
    return launch,observed,root,transport


def test_full_audit_with_real_preparation_and_substitute_runtime(tmp_path,inputs):
    args = fixture(tmp_path,inputs)
    result = audit(PREPARED,*args,PROOF,CPU)
    assert result['parity']['requests']==6 and result['parity']['exact']


@pytest.mark.parametrize('fault',['same_wrong_tensor','same_wrong_positions','same_wrong_panel'])
def test_backend_agreement_does_not_bypass_input_authentication(inputs,fault):
    _,config,menu,anchors,hf,vl = copy.deepcopy(inputs)
    for report in (hf,vl):
        if fault=='same_wrong_panel':
            report['input_panel'][0]['request_sha256']='bad'
            report['records'][0]['request_sha256']='bad'
        else:
            identity=report['records'][0]['request_identity']
            if fault=='same_wrong_tensor':identity['inputs']['input_features']['sha256']='bad'
            else:identity['positions'][0]+=1
            report['records'][0]['request_id']=content_hash(identity)
    assert compare(hf,vl)['exact']
    with pytest.raises(ValueError):bind_inputs(config,menu,anchors,hf,vl)


@pytest.mark.parametrize('fault',['device','mount','env','image','nonterminal','transport','terminal','profile'])
def test_runtime_or_artifact_mismatch_rejected(tmp_path,inputs,fault):
    launch,observed,root,transport=fixture(tmp_path,inputs)
    if fault=='device':observed['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci6'
    if fault=='mount':observed['HostConfig']['Binds'][1]=observed['HostConfig']['Binds'][1].removesuffix(':ro')
    if fault=='env':observed['Config']['Env']=['TMPDIR=/mnt/cephfs']
    if fault=='image':observed['Image']='wrong'
    if fault=='nonterminal':observed['State']['Status']='running'
    if fault=='transport':transport['files'].pop('evaluation/hf.json')
    if fault in ('terminal','profile'):
        name='evaluation/terminal.json' if fault=='terminal' else 'evaluation/hf-runtime.json'
        value={'exit_code':1,'source_manifest_sha256':'wrong'} if fault=='terminal' else {'profile':'wrong'}
        (root/name).write_text(json.dumps(value));transport['files'][name]=sha(root/name)
    with pytest.raises(ValueError):audit(PREPARED,launch,observed,root,transport,PROOF,CPU)
