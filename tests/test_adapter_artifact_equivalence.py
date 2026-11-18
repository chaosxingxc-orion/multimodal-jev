import json
import pytest
from multimodal_jev.policy_identity import checkpoint_identity, compare_adapter_artifacts


def pair(tmp_path):
    roots=[tmp_path/'a',tmp_path/'b']
    for i,p in enumerate(roots):
        p.mkdir()
        (p/'adapter_model.safetensors').write_bytes(b'identical-weights')
        (p/'adapter_config.json').write_text(json.dumps(dict(r=4,lora_alpha=8,
            target_modules=['q','v'] if i==0 else ['v','q'])))
    return roots


def test_target_set_order_receipt_keeps_original_ids_and_files(tmp_path):
    a,b=pair(tmp_path)
    before=[checkpoint_identity(p) for p in (a,b)]
    assert before[0][0]!=before[1][0]
    report=compare_adapter_artifacts(a,b)
    assert report['exact_weight_bytes']
    assert [report['reference_id'],report['candidate_id']]==[v[0] for v in before]
    assert before==[checkpoint_identity(p) for p in (a,b)]


@pytest.mark.parametrize('damage',['weight','rank','alpha','target','duplicate','regex','missing','extra','other_list'])
def test_reject_every_difference_except_module_set_order(tmp_path,damage):
    a,b=pair(tmp_path);p=b/'adapter_config.json';c=json.loads(p.read_text())
    if damage=='weight':(b/'adapter_model.safetensors').write_bytes(b'different')
    elif damage=='missing':(b/'adapter_model.safetensors').unlink()
    elif damage=='extra':(b/'extra.json').write_text('{}')
    elif damage=='rank':c['r']=8
    elif damage=='alpha':c['lora_alpha']=16
    elif damage=='target':c['target_modules']=['q','k']
    elif damage=='duplicate':c['target_modules']=['q','v','v']
    elif damage=='regex':c['target_modules']='q|v'
    else:
        other=json.loads((a/'adapter_config.json').read_text());other['layers_to_transform']=[0,1]
        (a/'adapter_config.json').write_text(json.dumps(other));c['layers_to_transform']=[1,0]
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):compare_adapter_artifacts(a,b)
