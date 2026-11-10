import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shards import make_plan,load_shard,population
from multimodal_jev.sfc_jev_shard_runtime import digest
from test_sfc_jev import load_tool
from test_sfc_jev_shards import fixture as input_fixture
from test_sfc_teacher_feedback import fixture as gold_fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def fixture(tmp_path,monkeypatch,part=None):
    module=load_tool('merge_sfc_jev_shards');release,partition=input_fixture(tmp_path)
    _,gold,_=gold_fixture();manifest=json.loads((release/'MANIFEST.json').read_text())
    for split in manifest['splits']:
        targets=[json.loads(json.dumps(t).replace('train_id',split)) for t in gold]
        # Rebind target hashes to each synthetic official input split.
        rows=population(release,split)
        for r,t in zip(rows,targets):
            t['request_sha256']=digest(r)
            native=dict(id=r['id'],split=r['split'],source_group=r['source_group'],input=r['input']['context'],audio_sha256=r['audio_sha256'])
            t['native_target']['request_sha256']=digest(native)
        path=release/f'{split}.targets.jsonl';path.write_bytes(b''.join(map(canonical,targets)))
        manifest['splits'][split]['targets_sha256']=file_hash(path)
    (release/'MANIFEST.json').write_bytes(canonical(manifest))
    registered=json.loads(partition.read_text());registered['manifest_sha256']=file_hash(release/'MANIFEST.json')
    partition.write_bytes(canonical(registered))
    kwargs=dict(partition_path=partition,partition_hash=file_hash(partition),part=part) if part else {}
    split='train_id' if part else 'test_id';plan=make_plan(release,split,12,**kwargs)
    path=tmp_path/'plan.json';path.write_bytes(canonical(plan));entries=[];receipts={}
    by_id={t['id']:t for t in map(json.loads,(release/f'{split}.targets.jsonl').read_text().splitlines())}
    for shard in plan['shards']:
        root=tmp_path/shard['id'];(root/'run').mkdir(parents=True)
        rows=load_shard(plan,shard['id']);predictions=[]
        for r in rows:
            predictions.extend(dict(example_id=r['id'],decision_id=d['id'],request_sha256=digest(r),
                raw_text=json.dumps(by_id[r['id']]['answers'][d['id']]),extra_evidence='retained') for d in r['input']['decisions'])
        cfg=dict(prompt_id='parent',shard_plan=str(path),shard_plan_sha256=file_hash(path),shard_id=shard['id'],split=split,
            instruction_sha256='parenthash',image='same-image',seed=17,max_new_tokens=1024)
        (root/'config.json').write_bytes(canonical(cfg));(root/'launch.json').write_text('{}')
        (root/'run/identity.json').write_bytes(canonical(dict(source_files={'runner':'hash'},model_files=['weights'],versions={'vllm':'pin'})))
        pred=root/'run/predictions.jsonl';pred.write_bytes(b''.join(map(canonical,predictions)))
        receipt=dict(shard_id=shard['id'],shard_plan_sha256=file_hash(path),split=split,
            bundles=shard['bundles'],decisions=shard['decisions'],source_groups=len(shard['source_groups']),
            prompt_id='parent',instruction_sha256='parenthash',container_id='cid-'+shard['id'],predictions_sha256=file_hash(pred))
        receipts[str(root.resolve())]=receipt
        entries.append(dict(shard_id=shard['id'],root=str(root),config_sha256=file_hash(root/'config.json'),launch_sha256=file_hash(root/'launch.json')))
    roster=tmp_path/'roster.json';roster.write_bytes(canonical(dict(schema='sfc-jev-completed-shard-roster-v1',plan_sha256=file_hash(path),runs=entries)))
    monkeypatch.setattr(module,'accept',lambda root:receipts[str(root.resolve())])
    return module,path,roster,release,receipts


def test_complete_merge_restores_release_order_retains_raw_fields_without_targets(tmp_path,monkeypatch):
    m,plan,roster,release,_=fixture(tmp_path,monkeypatch)
    for p in release.glob('*.targets.jsonl'):p.unlink()
    p,rows,ps,receipts,actor=m.collect(plan,file_hash(plan),roster,file_hash(roster))
    assert len(ps)==24 and len(rows)==4 and len(receipts)==2
    assert [(v['example_id'],v['decision_id']) for v in ps]==[(r['id'],d['id']) for r in rows for d in r['input']['decisions']]
    assert all(v['extra_evidence']=='retained' for v in ps)


@pytest.mark.parametrize('change',['missing_shard','duplicate_root','wrong_launch','mixed_seed','mixed_source',
    'wrong_population','duplicate_container','missing_decision','duplicate_decision','bad_binding','failed_acceptance'])
def test_incomplete_or_incompatible_shards_rejected_before_gold(tmp_path,monkeypatch,change):
    m,plan,roster,release,receipts=fixture(tmp_path,monkeypatch)
    data=json.loads(roster.read_text());second=data['runs'][1];root=Path(second['root'])
    if change=='missing_shard':data['runs'].pop()
    if change=='duplicate_root':second['root']=data['runs'][0]['root']
    if change=='wrong_launch':second['launch_sha256']='bad'
    if change in ('mixed_seed','wrong_population'):
        cfg=json.loads((root/'config.json').read_text());cfg['seed' if change=='mixed_seed' else 'split']='different'
        (root/'config.json').write_bytes(canonical(cfg));second['config_sha256']=file_hash(root/'config.json')
    if change=='mixed_source':
        p=root/'run/identity.json';identity=json.loads(p.read_text());identity['source_files']['runner']='changed';p.write_bytes(canonical(identity))
    if change=='duplicate_container':receipts[str(root)]['container_id']=next(iter(receipts.values()))['container_id']
    if change in ('missing_decision','duplicate_decision','bad_binding'):
        p=root/'run/predictions.jsonl';ps=list(map(json.loads,p.read_text().splitlines()))
        if change=='missing_decision':ps.pop()
        if change=='duplicate_decision':ps[1]=ps[0]
        if change=='bad_binding':ps[0]['request_sha256']='changed'
        p.write_bytes(b''.join(map(canonical,ps)));receipts[str(root)]['predictions_sha256']=file_hash(p)
    if change=='failed_acceptance':
        def fail(_):raise ValueError('actual container failed')
        monkeypatch.setattr(m,'accept',fail)
    roster.write_bytes(canonical(data))
    for p in release.glob('*.targets.jsonl'):p.unlink()
    scorer=tmp_path/'scorer';scorer.write_text('unused')
    with pytest.raises(ValueError):m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('part',[None,'learning','selection'])
def test_complete_scoring_uses_whole_named_population_and_original_gold(tmp_path,monkeypatch,part):
    m,plan,roster,release,_=fixture(tmp_path,monkeypatch,part);scorer=tmp_path/'scorer';scorer.write_text('synthetic')
    seen=[]
    def native(manifest,predictions,official,output,pin):
        rs=list(map(json.loads,manifest.read_text().splitlines()));ps=list(map(json.loads,predictions.read_text().splitlines()))
        assert len(rs)==len(ps)==(4 if part is None else 2)
        assert all('supervision' in r for r in rs)
        seen.extend(r['example_id'] for r in rs)
        return dict(scope='synthetic native evaluator stub')
    # Actual typed strict scoring and lossless native reconstruction execute.
    monkeypatch.setitem(m.evaluate.__globals__,'evaluate_native',native)
    result=m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),tmp_path/'output')
    assert result['bundles']==len(seen) and result['counts']['all_decisions_correct']==len(seen)
    assert result['decisions']==(24 if part is None else 12)
    assert (result['partition'] is None)==(part is None)
    if part is not None:
        assert result['partition']['part']==part
        assert (tmp_path/'output/scoring-population/train_id.targets.jsonl').exists()
    with pytest.raises(FileExistsError):m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),tmp_path/'output')


def test_bad_raw_output_counts_as_error_without_repair(tmp_path,monkeypatch):
    m,plan,roster,release,receipts=fixture(tmp_path,monkeypatch)
    root=Path(json.loads(roster.read_text())['runs'][0]['root']);p=root/'run/predictions.jsonl'
    rows=list(map(json.loads,p.read_text().splitlines()));rows[0]['raw_text']='```json\ntrue\n```'
    p.write_bytes(b''.join(map(canonical,rows)));receipts[str(root)]['predictions_sha256']=file_hash(p)
    monkeypatch.setitem(m.evaluate.__globals__,'evaluate_native',lambda *a:{})
    scorer=tmp_path/'scorer';scorer.write_text('synthetic')
    result=m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),tmp_path/'out')
    assert result['counts']['format_valid']==23
    merged=list(map(json.loads,(tmp_path/'out/predictions.jsonl').read_text().splitlines()))
    assert merged[0]['raw_text']=='```json\ntrue\n```'


def test_scoring_failure_retains_merge_and_terminal_without_replay(tmp_path,monkeypatch):
    m,plan,roster,release,_=fixture(tmp_path,monkeypatch)
    def failed(*args):raise RuntimeError('upstream scorer failure')
    monkeypatch.setitem(m.evaluate.__globals__,'evaluate_native',failed)
    scorer=tmp_path/'scorer';scorer.write_text('synthetic');out=tmp_path/'out'
    with pytest.raises(RuntimeError,match='upstream'):
        m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),out)
    assert json.loads((out/'terminal.json').read_text())['exit_code']==1
    assert (out/'predictions.jsonl').exists() and not (out/'result.json').exists()
    with pytest.raises(FileExistsError):m.score(plan,file_hash(plan),roster,file_hash(roster),scorer,file_hash(scorer),out)


@pytest.mark.parametrize('mixed',[False,True])
def test_merge_preserves_candidate_identity_and_rejects_mixed_prompts(tmp_path,monkeypatch,mixed):
    # Acceptance itself is stubbed here; the candidate wire/context gates are
    # tested separately. Exercise actual cross-shard merge identity checks.
    m,plan,roster,_,receipts=fixture(tmp_path,monkeypatch)
    data=json.loads(roster.read_text())
    for index,entry in enumerate(data['runs']):
        root=Path(entry['root']);cfg=json.loads((root/'config.json').read_text())
        prompt='p2' if mixed and index else 'p1'
        cfg.update(prompt_id=prompt,instruction_sha256=prompt+'hash',candidate={'pinned':prompt})
        (root/'config.json').write_bytes(canonical(cfg));entry['config_sha256']=file_hash(root/'config.json')
        receipts[str(root)].update(prompt_id=prompt,instruction_sha256=prompt+'hash')
    roster.write_bytes(canonical(data))
    if mixed:
        with pytest.raises(ValueError,match='mixed actor'):m.collect(plan,file_hash(plan),roster,file_hash(roster))
    else:
        _,_,records,_,actor=m.collect(plan,file_hash(plan),roster,file_hash(roster))
        assert len(records)==24 and actor['config']['prompt_id']=='p1'
