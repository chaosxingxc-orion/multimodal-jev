import copy
import hashlib
import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.direct_readout import alternatives
from multimodal_jev.frozen_contract import FrozenContract
from multimodal_jev.slurp_jev_full import make_bank, expand_bundle, values_from_target, reconstruct, build
from multimodal_jev.task_schema import DecisionSchema


def fixture():
    bank=make_bank({'scenario_action_pairs':[['email','send'],['music','play']],
                    'entity_types':['artist_name','person']})
    bundle={'id':'private-id','split':'train','source_group':'private-group','audio_sha256':'hash',
            'input':{'audio':'/audio','task_bank_sha256':hashlib.sha256(canonical(bank)).hexdigest()}}
    target={'scenario':'email','action':'send','entities':[{'type':'person','filler':'amy'}]*2}
    return bank,bundle,target


def test_multivalue_preserves_duplicates_and_distinguishes_empty_null():
    bank,bundle,target=fixture();values=values_from_target(target,bank)
    assert values['entity.person.values']==['amy','amy']
    assert values['entity.artist_name.values']==[]
    assert values['entity.person.present'] is True and values['entity.person.absent'] is False
    assert values['intent.native']!=values['intent.rebind']
    assert reconstruct(values,bank)==target
    schema=DecisionSchema.from_dict(bank['tasks']['entity.person.values'])
    assert schema.accepts([]) and schema.accepts(['amy','amy'])
    for bad in [None,'amy',[None],[''],[True],{'x':'amy'}]:assert not schema.accepts(bad)
    request=next(r for r in expand_bundle(bundle,bank) if r['id'].endswith('.values'))
    with pytest.raises(ValueError,match='open_slot'):alternatives(request)


def test_input_expansion_independent_of_gold_and_ids_not_rendered():
    bank,bundle,target=fixture();requests=list(expand_bundle(bundle,bank))
    changed=copy.deepcopy(target);changed['entities']=[];changed['scenario']='music';changed['action']='play'
    assert values_from_target(changed,bank)!=values_from_target(target,bank)
    assert requests==list(expand_bundle(bundle,bank)) and len(requests)==9
    contract=FrozenContract({'route':'audio-direct'})
    for request in requests:
        prompt=contract.audio_instruction(request,'Listen')
        assert 'private-id' not in prompt and 'private-group' not in prompt and 'amy' not in prompt
    poisoned=copy.deepcopy(bundle);poisoned['input']['gold']=target
    with pytest.raises(ValueError):list(expand_bundle(poisoned,bank))


def test_build_all_splits_and_reconstruct_complete_entities(tmp_path):
    bank,base,target=fixture();native=tmp_path/'native';native.mkdir()
    taxonomy={k:bank[k] for k in ('scenario_action_pairs','entity_types')}
    (native/'taxonomy.json').write_bytes(canonical(taxonomy))
    manifest={'identity':{'taxonomy_sha256':file_hash(native/'taxonomy.json')},'splits':{}}
    for split in ('train','valid','test'):
        request={**base,'id':split,'split':split,'source_group':split,'audio_sha256':split,
                 'input':{'audio':'/'+split,'schema_sha256':manifest['identity']['taxonomy_sha256']}}
        gold={'id':split,'target':target,'request_sha256':hashlib.sha256(canonical(request)).hexdigest(),
              'metadata':{'transcript_reference':'private reference'}}
        (native/f'{split}.requests.jsonl').write_bytes(canonical(request))
        (native/f'{split}.targets.jsonl').write_bytes(canonical(gold))
        manifest['splits'][split]={'recordings':1,**{k+'_sha256':file_hash(native/f'{split}.{k}.jsonl') for k in ('requests','targets')}}
    (native/'MANIFEST.json').write_bytes(canonical(manifest))
    output,receipt=build(native,tmp_path/'out',tmp_path/'state')
    assert receipt['native_entity_multiset_parity'] and receipt['decisions_per_recording']==9
    for split in ('train','valid','test'):
        assert receipt['splits'][split]['multiple_occurrence_queries']==1
        bundle=json.loads((output/f'{split}.bundles.jsonl').read_text())
        gold=json.loads((output/f'{split}.targets.jsonl').read_text())
        assert 'private reference' not in json.dumps(bundle)
        assert gold['bundle_sha256']==hashlib.sha256(canonical(bundle)).hexdigest()
        assert reconstruct(gold['values'],bank)==target
    with pytest.raises(FileExistsError):build(native,tmp_path/'out',tmp_path/'state')
