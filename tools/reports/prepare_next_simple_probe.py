"""Prepare one paired TRAIN probe of the registered layout; no model or device."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from multimodal_jev.baseline import file_hash
from multimodal_jev.boundary_panel import rank
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import ApplicabilityQuestion, ContextDecisionRequest, PREFIX_LAYOUT, render_prefix_context_request
from multimodal_jev.local_checkpoint import commit_json, sync_directory
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.stage1_candidate_layout import training_request
from tools.reports.build_jev_boundary_panel import source_rows
from tools.reports.jev_train_confusions import pinned


def render_panel(request):
    system, _, readouts = render_prefix_context_request(request)
    return dict(system=system, readouts=readouts, questions=[asdict(q) for q in request.questions],
                requirements=list(request.requirements), version=request.version,
                candidate_ids=[q.id for q in request.questions], prompt_layout=PREFIX_LAYOUT, history=[])


def unseen(request, conf, evaluation):
    if conf['layouts'] != ['native', 'both'] or evaluation != {'comparison': 'unseen_order', 'seed_offset': 1}:
        raise ValueError('registered evaluation-only permutation required')
    seed = conf['permutation_seed'] + evaluation['seed_offset']
    prior = {training_request(request, layout=arm, seed=conf['permutation_seed']).questions
             for arm in conf['layouts']}
    novel = training_request(request, layout='both', seed=seed)
    if novel.questions in prior:
        raise ValueError('evaluation order collides with trained order')
    return novel, dict(evaluation, seed=seed)


def full_train(c, provenance):
    out = Path(c['output'])
    check_paths({'seal': out})
    if out.exists():
        raise FileExistsError(out)
    design = json.loads(pinned(Path(c['design']['path']), c['design']['sha256']))
    plan = json.loads(pinned(Path(c['layout_plan']['path']), c['layout_plan']['sha256']))
    if (design['schema'] != 'jev-next-simple-two-stage-v1'
            or plan['design_sha256'] != c['design']['sha256'] or plan['source_identity'] != c['clean']['sha256']
            or plan['stage1_design_sha256'] != design['stage1']['sha256']):
        raise ValueError('layout plan must match this round and TRAIN release')
    task, rows = source_rows(c)
    request = compile_candidates('audio', [dict(id=q['id'], definition=q['description'])
        for q in task['schema']['candidates']], task['thinking']['requirements'], semantics='exclusive_event')
    variant, protocol = unseen(request, design['training'], c['evaluation'])
    for arm in design['training']['layouts']:
        trained = render_panel(training_request(request, layout=arm, seed=design['training']['permutation_seed']))
        plan_panel = plan['panels'][arm]
        if (trained['candidate_ids'], trained['system'], trained['readouts']) != (
                plan_panel['readout_ids'], plan_panel['system'], plan_panel['readouts']):
            raise ValueError('recompiled template differs from this round')

    rows.sort(key=lambda r: r['id'])
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique complete TRAIN roster required')
    population = len(rows)
    sampling = c.get('sampling')
    if sampling is not None:
        if (type(sampling['per_intent']) is not int or sampling['per_intent'] < 1
                or type(sampling['seed']) is not int or sampling['seed'] < 0):
            raise ValueError('explicit positive paired-panel quota and fixed seed required')
        by_intent = defaultdict(list)
        for row in rows:
            by_intent[row['gold']].append(row)
        rows = sorted((row for group in by_intent.values()
                       for row in sorted(group, key=lambda r: (rank(sampling['seed'], r['id']), r['id']))
                       [:sampling['per_intent']]), key=lambda r: r['id'])
    spec = dict(recordings=len(rows), candidates=len(request.questions), split='source_train_diagnostic')
    roster = [{k: r[k] for k in ('id', 'audio', 'audio_sha256')} for r in rows]
    targets = [{k: r[k] for k in ('id', 'audio_sha256', 'gold', 'group', 'speaker')} | {'stratum': r['gold']}
               for r in rows]
    ids = {q.id for q in request.questions}
    if any(t['gold'] not in ids or not t['group'] for t in targets):
        raise ValueError('trusted label and source group required')
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root = Path(tmp) / 'seal'
        root.mkdir()
        files = {}

        def write(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            commit_json(path, value)
            files[name] = file_hash(path)

        for arm, req in (('native', request), ('unseen', variant)):
            write(f'inference/{arm}/panel.json', render_panel(req))
            write(f'inference/{arm}/rows.json', roster)
        write('scoring/targets.json', targets)
        write('scoring/pairs.json', [])
        commit_json(root / 'MANIFEST.json', dict(schema='jev-native-evaluation-seal-v1',
            prompt_layout=PREFIX_LAYOUT, datasets={arm: spec for arm in ('native', 'unseen')},
            target_tuning_allowed=False, threshold=.5, files=files, diagnostic='full_train_unseen_layout',
            design_sha256=c['design']['sha256'], parent_sha256=c['layout_plan']['sha256'],
            clean_sha256=c['clean']['sha256'], native_sha256=c['native']['sha256'],
            evaluation=protocol, provenance=provenance,
            **(dict(sampling=sampling, source_recordings=population) if sampling is not None else {})))
        root.rename(out)
        sync_directory(out.parent)
    return dict(**spec, seal_sha256=file_hash(out / 'MANIFEST.json'), evaluation=protocol,
                model_calls=0, npu_executed=False, observed_errors=None)


def run(c):
    out=Path(c['output']);check_paths({'seal':out})
    parent=Path(c['parent']['root']);m=json.loads(pinned(parent/'MANIFEST.json',c['parent']['sha256']))
    def read(n):return json.loads(pinned(parent/n,m['files'][n]))
    d=json.loads(pinned(Path(c['design']['path']),c['design']['sha256']))
    conf=d['training'];p=read('inference/original/panel.json');rows=read('inference/original/rows.json')
    spec=m['datasets']['original']
    if (out.exists() or spec['split']!='source_train_diagnostic' or conf['layouts']!=['native','both']
            or m['target_tuning_allowed'] is not False):raise ValueError('fresh full TRAIN mechanism probe required')
    req=ContextDecisionRequest('audio',tuple(ApplicabilityQuestion(**q) for q in p['questions']),
        tuple(p['requirements']),tuple(p['history']),p['version'])
    evaluation=c.get('evaluation')
    orders={arm:(arm,conf['permutation_seed']) for arm in conf['layouts']}
    if evaluation is not None:
        _, protocol = unseen(req, conf, evaluation)
        orders={'native':orders['native'],'unseen':('both',protocol['seed'])}
    out.parent.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        root=Path(tmp)/'seal';root.mkdir();files={}
        def write(n,obj):
            f=root/n;f.parent.mkdir(parents=True,exist_ok=True);commit_json(f,obj);files[n]=file_hash(f)
        for k,(arm,seed) in orders.items():
            request=training_request(req,layout=arm,seed=seed)
            system,_,readouts=render_prefix_context_request(request)
            panel=dict(p,system=system,readouts=readouts,questions=[asdict(q) for q in request.questions],
                       candidate_ids=[q.id for q in request.questions])
            if k=='native' and p!=panel:raise ValueError('native rendering differs')
            write(f'inference/{k}/panel.json',panel);write(f'inference/{k}/rows.json',rows)
        for n in ('scoring/targets.json','scoring/pairs.json'):write(n,read(n))
        extra={} if evaluation is None else dict(evaluation=protocol)
        commit_json(root/'MANIFEST.json',dict(m,**extra,files=files,datasets={k:spec for k in orders},
            design_sha256=c['design']['sha256'],parent_sha256=c['parent']['sha256']))
        root.rename(out)
    return file_hash(out/'MANIFEST.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args()
    c, provenance = load_config(a.config, a.local)
    if c.get('schema') == 'jev-full-train-unseen-probe-v1':
        print(json.dumps(full_train(c, provenance), indent=2))
    else:
        print(run(c))
