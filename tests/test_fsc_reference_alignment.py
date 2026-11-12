import copy
import pytest
from multimodal_jev.fsc_reference_alignment import align


def fixture():
    source = dict(id='fsc:wavs/a.wav',audio='/data/wavs/a.wav',audio_sha256='a'*64,speaker_id='s')
    pop = dict(schema='fsc-official-span-population-v1',split='test',role='frozen_transfer_evaluation_inputs',
        dataset_root='/data',records=[source],recordings=1,speakers=1)
    pred = dict(example_id='wavs/a.wav',audio_sha256='a'*64,raw_prediction='unparsed output retained')
    return pop,[pred]


def test_alignment_does_not_require_labels_or_parse_output():
    pop,pred=fixture(); result=align(pop,pred,verify_audio=False)
    assert result['passed'] and result['recordings']==1


@pytest.mark.parametrize('kind',['missing','extra','duplicate','audio','id','outside'])
def test_mismatched_reference_rejected(kind):
    pop,pred=fixture()
    if kind=='missing':pred=[]
    if kind=='duplicate':pred*=2
    if kind=='extra':pred.append(dict(pred[0],example_id='extra.wav'))
    if kind=='audio':pred[0]['audio_sha256']='b'*64
    if kind=='id':pop['records'][0]['id']='fsc:wrong'
    if kind=='outside':pop['records'][0]['audio']='/elsewhere/a.wav'
    with pytest.raises(ValueError):align(pop,pred,verify_audio=False)
