import hashlib
import pytest
from multimodal_jev.audio2tool_inventory import audio_inventory, registry, REVISION


def row(tier,query,name):
    path=f'public/tier{tier}_example_audios/query_{query:05d}/{name}'
    return dict(asset='audio2tool',path='datasets/audio2tool/'+path,
        url=f'https://huggingface.co/datasets/RVtech/Audio2Tool/resolve/{REVISION}/{path}',
        algorithm='sha256',hash='a'*64,size=100)


def test_numeric_conversation_order_preserves_roles_and_separates_variants():
    files=[row(7,1,'turn_10_user_a.wav'),row(7,1,'turn_2_agent_b.wav'),
           row(7,1,'turn_1_user_a.wav'),row(1,1,'speaker_01_x.wav'),row(1,1,'speaker_00_y.wav')]
    result=audio_inventory(files)
    assert result[0]['audio_semantics']=='independent_variants'
    assert len(result[0]['audios'])==2
    assert result[1]['audio_semantics']=='ordered_conversation_turns'
    assert [(a['turn'],a['role']) for a in result[1]['audios']]==[(1,'user'),(2,'agent'),(10,'user')]


def test_ambiguous_duplicate_turn_fails_instead_of_guessing():
    with pytest.raises(ValueError,match='duplicate conversation'):
        audio_inventory([row(7,1,'turn_1_user_a.wav'),row(7,1,'turn_1_agent_b.wav')])


@pytest.mark.parametrize('field,value', [('path','/tmp/file.wav'),('algorithm','git-sha1'),
    ('hash','invalid'),('url','https://example.com/a.wav'),('size',0)])
def test_bad_asset_identity_rejected(field,value):
    value_row=row(1,0,'speaker_00_x.wav');value_row[field]=value
    with pytest.raises(ValueError):audio_inventory([value_row])


def test_duplicate_manifest_path_rejected():
    r=row(1,0,'speaker_00_x.wav')
    with pytest.raises(ValueError,match='duplicate asset'):audio_inventory([r,r])


def test_registry_preserves_definition_order_and_defaults_without_labels():
    data=b'tool_id,domain,category,tool_name,signature,description,argument_defaults,argument_constraints\n2,home,x,z,z(),last,none,none\n1,car,x,a,a(),first,none,none\n'
    result=registry(data,hashlib.sha256(data).hexdigest())
    assert [r['tool_name'] for r in result]==['z','a']
    assert result[0]['argument_defaults']=='none'
    with pytest.raises(ValueError,match='hash'):registry(data,'0'*64)
    duplicate=data+data.splitlines(keepends=True)[1]
    with pytest.raises(ValueError,match='duplicate'):registry(duplicate,hashlib.sha256(duplicate).hexdigest())


def test_inventory_writer_authenticates_registry_against_manifest(tmp_path):
    import json
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    from prepare_audio2tool_inventory import run
    data=b'tool_id,domain,category,tool_name,signature,description,argument_defaults,argument_constraints\n0,home,x,a,a(),description,,\n'
    path=tmp_path/'registry.csv';path.write_bytes(data)
    pin=dict(asset='audio2tool',path='datasets/audio2tool/tools_registry.csv',
        url=f'https://huggingface.co/datasets/RVtech/Audio2Tool/resolve/{REVISION}/tools_registry.csv',
        algorithm='git-sha1',hash=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),size=len(data))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(assets=[dict(asset='audio2tool',repo='RVtech/Audio2Tool',revision=REVISION,files=2)],
        files=[pin,row(8,0,'mixed.wav')])))
    result=run(manifest,path,tmp_path/'output')
    assert result['tools']==1 and result['inventory_audio_files']==1
    with pytest.raises(FileExistsError):run(manifest,path,tmp_path/'output')
    path.write_bytes(data.replace(b'description',b'changed'))
    with pytest.raises(ValueError,match='registry differs'):run(manifest,path,tmp_path/'tampered')
    assert not (tmp_path/'tampered').exists()
