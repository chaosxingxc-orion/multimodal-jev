import json
import pytest
from multimodal_jev.sfc import parse_calls, strict_metrics, model_input, source_identity


def calls(*values):
    return json.dumps(values)


def test_sfc_parser_preserves_arrays_and_missing_sentinel():
    result=parse_calls(calls('find(time_range=["Monday", "Friday"], n=2, person="NAN")'))
    assert result[0]['arguments']=={'time_range':['Monday','Friday'],'n':2,'person':'NAN'}
    for text in [calls('f(x=do_something())'),calls('obj.f(x="a")'),calls('f("a")'),
                 calls('f(x="a",x="b")'),calls('f(**{})'),calls('f(x=1e999)'),
                 'f(x="a")','[null]']:
        with pytest.raises(ValueError):parse_calls(text)


def test_strict_sfc_metrics_preserve_function_slot_binding_and_duplicates():
    gold=calls('a(x="one")','b(x="two")')
    swapped=calls('a(x="two")','b(x="one")')
    scores=strict_metrics([swapped],[gold])
    assert scores['function_multiset_correct']==1 and scores['exact_correct']==0
    assert scores['bound_slot_f1']==0
    assert strict_metrics([calls('a(x="one")','a(x="one")')],[calls('a(x="one")')])['exact_correct']==0
    assert strict_metrics(['bad'],['[]'])['exact_correct']==0
    assert strict_metrics(['[]'],['[]'])['exact_correct']==1
    assert strict_metrics([calls('a(x=["Monday"])')],[calls('a(x=["Friday"])')])['exact_correct']==0


def test_sfc_input_preserves_two_audio_turns_but_excludes_labels():
    paths=['audio/final_data_audio/level_3_1/test_ood/id_7_1.wav',
           'audio/final_data_audio/level_3_1/test_ood/id_7_2.wav']
    record={'messages':[{'role':'user','content':'<audio>'},{'role':'assistant','content':'clarify'},
                        {'role':'user','content':'<audio>'}], 'audios':paths,'solution':'SECRET','target':'SECRET'}
    inputs=model_input(record)
    assert set(inputs)=={'messages','audios'} and inputs['audios']==paths
    assert 'SECRET' not in json.dumps(inputs)
    assert source_identity(paths)['source_group']==source_identity(paths[:1])['source_group']
    assert source_identity(paths)['stage']==2
    inputs['messages'][0]['content']='changed'
    assert record['messages'][0]['content']=='<audio>'


def test_sfc_release_preserves_original_splits_inputs_and_labels(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('sfc_builder',Path(__file__).parents[1]/'tools/jev/build_sfc_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'source'
    for split in ('train_id','test_id','test_ood'):
        audio=f'audio/final_data_audio/level_1/{split}/id_1.wav'
        path=root/'data'/audio;path.parent.mkdir(parents=True);path.write_bytes(split.encode())
        row={'messages':[{'role':'user','content':'schema <audio>'}],'audios':[audio],
             'solution':calls('find(x=["one","two"], person="NAN")')}
        data=root/'data/processed/data_msswift';data.mkdir(parents=True,exist_ok=True)
        (data/f'{split}.json').write_text(json.dumps([row]))
    release,receipt=module.build(root,tmp_path/'releases')
    for split in ('train_id','test_id','test_ood'):
        record=json.loads((release/f'{split}.jsonl').read_text())
        assert record['metadata']['official_split']==split
        assert record['supervision']['official_solution']==row['solution']
        assert record['input']['messages']==row['messages']
        assert set(record['input'])=={'audios','messages'}
    with pytest.raises(FileExistsError):module.build(root,tmp_path/'releases')
