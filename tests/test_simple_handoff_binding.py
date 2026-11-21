import json
from pathlib import Path

import pytest

from tools.reports import bind_simple_handoff as bind
from tools.reports.compile_simple_experiment import ROOT, compile_recipe


def current_predecessor(tmp_path):
    recipe = bind.read(ROOT/'configs/experiments/simple-two-stage-recipe.json')
    manifest = bind.binding(ROOT/recipe['manifest_template']['path'])
    design = bind.binding(ROOT/recipe['base']['path'])
    package = dict(design=design['path'],stage1_layout=dict(design=manifest,plan=dict(sha256='p')),
                   layout_cache=dict(sha256='c'))
    pipe = dict(design_sha256=design['sha256'],output=str(tmp_path/'pipeline'))
    final = dict(operations_design=design['path'],pipeline=dict(root=pipe['output'],
        expected_stage1_layout=dict(design=manifest['sha256'],plan='p',cache='c')))
    return recipe,package,pipe,final


@pytest.mark.parametrize('change', ['none','rank','device_count','predecessor','sealed_design'])
def test_predecessor_cannot_be_relabelled_as_changed_recipe(tmp_path, change):
    recipe,pkg,pipe,final = current_predecessor(tmp_path)
    if change == 'rank':
        for name in ('stage1_lora','stage2_lora'):
            recipe[name] = dict(rank=16,alpha=32)
    elif change == 'device_count':
        recipe['total_devices'] = 4
    elif change == 'predecessor':
        final['pipeline']['root'] = str(tmp_path/'other')
    elif change == 'sealed_design':
        other = tmp_path/'design.json'
        other.write_text('{}')
        final['operations_design'] = str(other)
    files = compile_recipe(recipe,Path('configs/experiments/example'))
    if change == 'none':
        bind.check_recipe(pkg,pipe,recipe,files,final)
    else:
        with pytest.raises(ValueError):
            bind.check_recipe(pkg,pipe,recipe,files,final)


def test_compiled_design_and_changed_adapter_can_be_bound_in_new_attempt(tmp_path):
    recipe, pkg, pipe, final = current_predecessor(tmp_path)
    recipe.update(total_devices=4, stage1_lora=dict(rank=16,alpha=32), stage2_lora=dict(rank=16,alpha=32))
    files = compile_recipe(recipe,tmp_path)
    for name,data in files.items():
        (tmp_path/name).write_bytes(data)
    pkg['stage1_layout']['design'] = manifest = bind.binding(tmp_path/'experiment.json')
    pipe['design_sha256'] = bind.binding(tmp_path/'two-stage.json')['sha256']
    final['pipeline']['expected_stage1_layout']['design'] = manifest['sha256']
    final['operations_design'] = pkg['design'] = str(tmp_path/'two-stage.json')
    design,_ = bind.check_recipe(pkg,pipe,recipe,files,final)
    batch = bind.reference_batch(design)
    assert design['stage1']['lora']['rank'] == 16
    plan = bind.device_plan([6,3,5,2],batch,{'b':'b.json','a':'a.json'})
    assert plan == dict(training_devices=[6,3],inference_services=[dict(name='a',device=5),dict(name='b',device=2)])
    with pytest.raises(ValueError,match='inventory'):
        bind.device_plan([6,6,5,2],batch,{'a':'a.json','b':'b.json'})


def test_publication_reuses_identical_file_never_replaces_competing_attempt(tmp_path,monkeypatch):
    monkeypatch.setattr(bind,'check_paths',lambda _:None)
    path = tmp_path/'handoff.json'
    bind.publish(path,{'run':'a'})
    inode = path.stat().st_ino
    bind.publish(path,{'run':'a'})
    assert inode == path.stat().st_ino
    with pytest.raises(FileExistsError):
        bind.publish(path,{'run':'b'})
    assert json.loads(path.read_bytes()) == {'run':'a'}


def test_public_stage1_entry_uses_layout_bound_profile_for_eight_cards(tmp_path):
    from tools.reports.run_simple_experiment import stage1_entry
    _,pkg,pipe,_ = current_predecessor(tmp_path)
    pkg.update(physical_devices=list(range(8)),package='finite',remote_root='/remote')
    pipe.update(package=pkg,finite={k:pkg[k] for k in ('package','remote_root')})
    design = bind.read(pkg['design'])
    stage1_entry(pipe,design)
    pkg['physical_devices'][-1] = pkg['physical_devices'][0]
    with pytest.raises(ValueError,match='topology'):
        stage1_entry(pipe,design)
