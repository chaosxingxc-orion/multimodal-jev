import copy
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('fsc_processor',Path(__file__).resolve().parents[1]/'tools/jev/preflight_fsc_real_audio.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def population():
    return dict(role='frozen_transfer_evaluation_inputs',schema='fsc-official-span-population-v1',
        split='test',recordings=2,speakers=1,records=[dict(id=str(i),audio=f'{i}.wav',audio_sha256='hash',speaker_id='speaker') for i in range(2)])


def test_published_evaluation_roster_supported_without_labels():
    value=population();before=copy.deepcopy(value)
    assert module.validate_roster(value)=='frozen_transfer_evaluation_inputs'
    assert value==before


@pytest.mark.parametrize('bad',['count','speakers','duplicate','label','split','role','empty'])
def test_rejects_invalid_or_label_bearing_population(bad):
    value=population()
    if bad=='count':value['recordings']=3
    elif bad=='speakers':value['speakers']=2
    elif bad=='duplicate':value['records'][1]=dict(value['records'][0])
    elif bad=='label':value['records'][0]['gold']='secret'
    elif bad=='split':value['split']='train'
    elif bad=='role':value['role']='unreviewed'
    else:value.update(records=[],recordings=0,speakers=0)
    with pytest.raises(ValueError):module.validate_roster(value)


def test_engineering_twelve_record_contract_preserved():
    value=dict(role='engineering_train_inputs',records=[dict(id=str(i),audio=f'{i}.wav') for i in range(12)])
    assert module.validate_roster(value)=='engineering_train_inputs'
    value['records'].pop()
    with pytest.raises(ValueError,match='twelve'):module.validate_roster(value)
