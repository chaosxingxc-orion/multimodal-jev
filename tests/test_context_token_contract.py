import copy,json
import pytest
from multimodal_jev.context_token_contract import validate_processor_extension
from multimodal_jev.context_decision_contract import DECISION_TOKEN,RESERVED_TIMESTAMP_TOKEN


def fixture(tmp_path):
    base=tmp_path/'base';proc=tmp_path/'processor';base.mkdir();proc.mkdir()
    old=dict(model=dict(vocab={'x':0,'last':151704}),normalizer=None,added_tokens=[])
    extended=copy.deepcopy(old)
    for i,text in ((151705,RESERVED_TIMESTAMP_TOKEN),(151706,DECISION_TOKEN)):
        extended['added_tokens'].append(dict(id=i,content=text,special=True,normalized=False,
            single_word=False,lstrip=False,rstrip=False))
    config=dict(timestamp_token_id=151705,text_config=dict(vocab_size=151936))
    for p,d in ((base/'tokenizer.json',old),(base/'config.json',config),(proc/'tokenizer.json',extended)):
        p.write_text(json.dumps(d))
    return base,proc,extended,config


def test_exact_extension(tmp_path):
    base,proc,_,_=fixture(tmp_path)
    assert validate_processor_extension(base,proc)['decision_id']==151706


@pytest.mark.parametrize('fault',['normalizer','old_vocab','duplicate','attributes','collision','capacity'])
def test_changed_contract_rejected(tmp_path,fault):
    base,proc,extended,config=fixture(tmp_path)
    if fault=='normalizer':extended['normalizer']='changed'
    if fault=='old_vocab':extended['model']['vocab']['x']=1
    if fault=='duplicate':extended['added_tokens'].append(extended['added_tokens'][0])
    if fault=='attributes':extended['added_tokens'][1]['normalized']=True
    if fault=='collision':config['another_token_id']=151706
    if fault=='capacity':config['text_config']['vocab_size']=151706
    (proc/'tokenizer.json').write_text(json.dumps(extended));(base/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError):validate_processor_extension(base,proc)
