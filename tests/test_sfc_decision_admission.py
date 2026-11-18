import json
from pathlib import Path

import pytest

from test_sfc_consumption import record
from multimodal_jev.sfc_consumption import SPLITS,split_record
from multimodal_jev.consumption import canonical
from tools.reports import audit_sfc_decision_admission as module


def setup(tmp_path):
    root=tmp_path/'native';root.mkdir();splits={}
    function=dict(name='f',description='A test function',parameters=dict(type='object',properties={}))
    prompt='### Available Functions:\n'+json.dumps([function])+"\n\n### Environment:\n{}\n\n### User's Audio Request:\n<audio>"
    for split in SPLITS:
        requests=[];targets=[]
        for stage in (1,2):
            row=record(split,stage);row['input']['messages'][0]['content']=prompt
            req,target=split_record(row);requests.append(req);targets.append(target)
        request_path=root/f'{split}.requests.jsonl';request_path.write_bytes(b''.join(map(canonical,requests)))
        # Deliberately invalid held-out files prove they are never parsed or hashed.
        target_path=root/f'{split}.targets.jsonl'
        target_path.write_bytes(b''.join(map(canonical,targets)) if split=='train_id' else b'not gold\n')
        splits[split]=dict(requests=2,source_groups=1,requests_sha256=module.file_hash(request_path),
            targets_sha256=module.file_hash(target_path))
    manifest=root/'MANIFEST.json';manifest.write_text(json.dumps(dict(identity={'dataset':'sfc-native'},splits=splits)))
    return dict(release=str(root),manifest_sha256=module.file_hash(manifest))


def test_all_inputs_but_only_train_labels_and_no_semantic_completeness_claim(tmp_path,monkeypatch):
    c=setup(tmp_path);original=Path.open
    def guarded(self,*args,**kwargs):
        if self.name in ('test_id.targets.jsonl','test_ood.targets.jsonl'):
            pytest.fail('held-out labels accessed')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result=module.audit(c)
    assert result['train_projection']['repeated_function_queries']==2
    assert result['train_projection']['missing_argument_NAN_queries']==2
    assert result['train_projection']['positive_memberships']==2
    assert result['splits']['test_id']['supplied_assistant_history']==1
    assert not result['heldout_targets_opened'] and not result['semantic_completeness_verified']
    assert not result['evaluation_dispatch_allowed'] and not result['applicability_claim_allowed']
    assert result['input_function_inventory']['test_ood_defined_also_in_train']==1


@pytest.mark.parametrize('fault',['manifest','hash','count','prefix','duplicate','label_binding','unknown_function'])
def test_mutated_or_inconsistent_release_refused(tmp_path,fault):
    c=setup(tmp_path);root=Path(c['release']);manifest=json.loads((root/'MANIFEST.json').read_text())
    if fault=='manifest':c['manifest_sha256']='wrong'
    elif fault=='hash':(root/'test_id.requests.jsonl').write_text('changed')
    else:
        if fault=='count':manifest['splits']['test_id']['requests']=3
        if fault in ('prefix','duplicate'):
            p=root/'test_id.requests.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
            if fault=='prefix':rows[1]['audio_sha256'][0]='a'*64
            else:rows[1]=rows[0]
            p.write_bytes(b''.join(map(canonical,rows)));manifest['splits']['test_id']['requests_sha256']=module.file_hash(p)
        if fault in ('label_binding','unknown_function'):
            p=root/'train_id.targets.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
            if fault=='label_binding':rows[0]['request_sha256']='wrong'
            else:
                rows[0]['supervision']['official_solution']='["unknown()"]'
                rows[0]['supervision']['calls']=[dict(name='unknown',arguments={})]
            p.write_bytes(b''.join(map(canonical,rows)));manifest['splits']['train_id']['targets_sha256']=module.file_hash(p)
        (root/'MANIFEST.json').write_text(json.dumps(manifest));c['manifest_sha256']=module.file_hash(root/'MANIFEST.json')
    with pytest.raises(ValueError):module.audit(c)
