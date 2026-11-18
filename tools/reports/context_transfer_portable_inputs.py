"""Portable full native evaluation inputs bound to independent population receipts.

No gold, policy selection or inference. Preserve logical audio paths/cache keys;
actual cache location is a local macro. Runtime verifies all request/rendering/
tensor identities against a separately pinned full-population evidence bundle.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ContextDecisionRequest, ApplicabilityQuestion
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_transfer_macro_inputs import load as load_original, normalize

SIZES={'FSC_native31':(3793,31),'MINDS_native14':(8168,14),'SLURP_clean_TEST60':(12991,60)}
EVIDENCE_FILES={'population.json','candidate-ids.json','input-receipts.json'}


def read(root,name,expected):
    p=Path(root)/name
    if p.is_symlink() or not p.resolve().is_relative_to(Path(root).resolve()) or file_hash(p)!=expected:
        raise ValueError('portable full input artifact differs: '+name)
    return json.loads(p.read_text())


def evidence(spec):
    root=Path(spec['root']);m=read(root,'MANIFEST.json',spec['manifest_sha256'])
    if (m['dataset'] not in SIZES or set(m['artifacts'])!=EVIDENCE_FILES
            or (m['recordings'],m['decisions'])!=SIZES[m['dataset']]):
        raise ValueError('independent full native population evidence required')
    values={n:read(root,n,h) for n,h in m['artifacts'].items()}
    return m,values


def prepare(c,provenance):
    admitted=load_original(c['evaluation'])
    m,values=evidence(c['population_evidence'])
    if (m['dataset']!=admitted['dataset'] or m['processor_sha256']!=admitted['processor_sha256']
            or values['population.json']!=admitted['population']
            or values['candidate-ids.json']!=admitted['candidate_ids']
            or values['input-receipts.json']!=admitted['input_receipts']):
        raise ValueError('original dataset admission differs from full population evidence')
    source_release=c['evaluation'].get('source_release') if m['dataset']=='SLURP_clean_TEST60' else None
    if m.get('source_release')!=source_release:
        raise ValueError('source held-out ontology release differs')
    requests=[];rows=[];receipts=[]
    for i,asset in enumerate(admitted['population']):
        requests.append(dict(id=asset['id'],request=admitted['requests'][asset['id']].model_input()))
        rows.append(dict(asset,rendered_sha256=admitted['source_rows'][i]['rendered_sha256']))
        prior=admitted['cache_receipts'][i]
        r={k:prior[k] for k in ('id','audio_sha256','request_sha256','cache_key','cache_sha256','positions','tensors')}
        if 'audio_receipt' in prior:r['audio_receipt']=prior['audio_receipt']
        receipts.append(r)
    out=Path(c['output']);check_paths({'portable_full_inputs':out});out.mkdir(parents=True,exist_ok=False)
    (out/'population-evidence').mkdir()
    # Preserve independent bytes exactly so relocation cannot replace their hash.
    for name in ['MANIFEST.json',*sorted(EVIDENCE_FILES)]:
        (out/'population-evidence'/name).write_bytes((Path(c['population_evidence']['root'])/name).read_bytes())
    for name,value in [('requests.json',requests),('rows.json',rows),('cache-receipts.json',receipts)]:
        commit_json(out/name,value)
    manifest=dict(schema='jev-portable-full-transfer-inputs-v1',dataset=m['dataset'],
        recordings=m['recordings'],decisions=m['decisions'],processor_sha256=m['processor_sha256'],
        population_evidence_sha256=c['population_evidence']['manifest_sha256'],
        files={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()},
        source_release=source_release,provenance=provenance,target_labels_included=False,
        full_population_inputs=True,cache_bytes_verified=False,model_execution_verified=False)
    commit_json(out/'MANIFEST.json',manifest)
    spec=dict(root=str(out),manifest_sha256=file_hash(out/'MANIFEST.json'),
        population_evidence_sha256=c['population_evidence']['manifest_sha256'],cache_root=admitted['cache_root'])
    load(spec)
    return dict(spec,dataset=m['dataset'],recordings=len(rows),decisions=m['decisions'],
        source_release=source_release,npu_executed=False)


def load(c):
    root=Path(c['root']);m=read(root,'MANIFEST.json',c['manifest_sha256'])
    files={'requests.json','rows.json','cache-receipts.json','population-evidence/MANIFEST.json',
           *{'population-evidence/'+n for n in EVIDENCE_FILES}}
    if (m['schema']!='jev-portable-full-transfer-inputs-v1' or m['dataset'] not in SIZES
            or set(m['files'])!=files or m['target_labels_included'] is not False
            or m['full_population_inputs'] is not True
            or m['population_evidence_sha256']!=c['population_evidence_sha256']
            or m['files']['population-evidence/MANIFEST.json']!=c['population_evidence_sha256']):
        raise ValueError('separately bound full-population input contract required')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=files|{'MANIFEST.json'} or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('exact label-free portable inventory required')
    values={n:read(root,n,h) for n,h in m['files'].items()}
    independent,expected=evidence(dict(root=str(root/'population-evidence'),
        manifest_sha256=c['population_evidence_sha256']))
    if any(m[k]!=independent[k] for k in ('dataset','recordings','decisions','processor_sha256')):
        raise ValueError('full target population metadata differs')
    if m['source_release']!=independent.get('source_release'):
        raise ValueError('portable source ontology release differs')
    rows=values['rows.json'];raw=values['requests.json'];receipts=values['cache-receipts.json'];requests=[]
    if len(rows)!=m['recordings'] or len(raw)!=len(rows):raise ValueError('full population required')
    required_receipt={'id','audio_sha256','request_sha256','cache_key','cache_sha256','positions','tensors'}
    if any(not required_receipt<=set(r) or set(r)-required_receipt-{'audio_receipt'} for r in receipts):
        raise ValueError('only label-free cache receipts allowed')
    for row,item in zip(rows,raw,strict=True):
        r=item['request']
        if (set(row)!={'id','audio_sha256','request_sha256','rendered_sha256'}
                or set(item)!={'id','request'} or item['id']!=row['id']
                or set(r)!={'audio','questions','requirements','history','version'}):
            raise ValueError('only label-free fixed native requests allowed')
        requests.append(ContextDecisionRequest(r['audio'],tuple(ApplicabilityQuestion(**q) for q in r['questions']),
            tuple(r['requirements']),tuple(r['history']),version=r['version']))
    candidates=expected['candidate-ids.json'];count,width=SIZES[m['dataset']]
    population,input_receipts=normalize(rows,receipts,requests,candidates,expected_count=count,decisions=width)
    if population!=expected['population.json'] or input_receipts!=expected['input-receipts.json']:
        raise ValueError('portable requests differ from independent full input evidence')
    return dict(dataset=m['dataset'],population=population,input_receipts=input_receipts,
        candidate_ids=candidates,requests={r['id']:q for r,q in zip(rows,requests,strict=True)},
        cache_receipts=receipts,source_rows=rows,processor_sha256=m['processor_sha256'],
        cache_root=c['cache_root'],source_release=m['source_release'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    print(json.dumps(prepare(config,provenance),indent=2))
