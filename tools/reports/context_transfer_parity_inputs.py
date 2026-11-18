"""Portable label-free parity panel inputs; not full target evaluation admission.

Prepare re-runs the full original dataset adapter and deterministic panel. The
runtime reader needs only this immutable bundle and an independently pinned
panel file hash. Original audio paths remain logical cache-key identities;
the actual cache root is supplied locally and raw audio is never opened here.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_decision_contract import ContextDecisionRequest,ApplicabilityQuestion
from multimodal_jev.context_macro_rollout import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.context_transfer_macro_inputs import load as load_full,normalize
from tools.reports.context_transfer_macro_parity import panel,WIDTHS


def prepare(c,provenance):
    admitted=load_full(c['evaluation']);selected=panel(admitted)
    expected=c['panel']
    if (set(expected)!={'path','sha256'} or file_hash(expected['path'])!=expected['sha256']
            or json.loads(Path(expected['path']).read_text())!=selected):
        raise ValueError('independently bound full-population panel differs')
    requests=[];receipts=[];rows=[]
    for i in selected['indices']:
        asset=admitted['population'][i];prior=admitted['cache_receipts'][i]
        raw=admitted['requests'][asset['id']].model_input()
        requests.append(dict(id=asset['id'],request=raw))
        row=dict(asset,rendered_sha256=admitted['source_rows'][i]['rendered_sha256'])
        rows.append(row)
        # Only fields consumed by the read-only cache verifier and normalizer.
        r={k:prior[k] for k in ('id','audio_sha256','request_sha256','cache_key','cache_sha256','positions','tensors')}
        if 'audio_receipt' in prior:r['audio_receipt']=prior['audio_receipt']
        receipts.append(r)
    out=Path(c['output']);check_paths({'parity_inputs':out});out.mkdir(parents=True,exist_ok=False)
    for name,value in [('panel.json',selected),('requests.json',requests),('rows.json',rows),('cache-receipts.json',receipts)]:
        commit_json(out/name,value)
    manifest=dict(schema='jev-portable-parity-inputs-v1',dataset=admitted['dataset'],
        panel_sha256=file_hash(out/'panel.json'),panel_content_sha256=digest(selected),
        processor_sha256=admitted['processor_sha256'],recordings=len(rows),
        files={p.name:file_hash(p) for p in out.iterdir() if p.is_file()},
        provenance=provenance,source_panel=expected,full_evaluation_allowed=False,
        cache_bytes_verified=False)
    commit_json(out/'MANIFEST.json',manifest)
    load(dict(root=str(out),manifest_sha256=file_hash(out/'MANIFEST.json'),panel_sha256=expected['sha256'],
        cache_root=admitted['cache_root']))
    return dict(output=str(out),manifest_sha256=file_hash(out/'MANIFEST.json'),recordings=len(rows),
        dataset=admitted['dataset'],full_evaluation_allowed=False,npu_executed=False)


def load(c):
    root=Path(c['root'])
    def read(name,expected):
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) or file_hash(p)!=expected:
            raise ValueError('portable parity artifact differs')
        return json.loads(p.read_text())
    m=read('MANIFEST.json',c['manifest_sha256'])
    if (m['schema']!='jev-portable-parity-inputs-v1' or m['dataset'] not in WIDTHS
            or m['full_evaluation_allowed'] is not False
            or set(m['files'])!={'panel.json','requests.json','rows.json','cache-receipts.json'}
            or m['panel_sha256']!=c['panel_sha256'] or m['files']['panel.json']!=c['panel_sha256']):
        raise ValueError('explicit parity-only input contract required')
    values={n:read(n,h) for n,h in m['files'].items()}
    selected=values['panel.json'];rows=values['rows.json'];receipts=values['cache-receipts.json']
    raw=values['requests.json'];requests=[]
    if (digest(selected)!=m['panel_content_sha256'] or selected['dataset']!=m['dataset']
            or selected['processor_sha256']!=m['processor_sha256']
            or len(raw)!=m['recordings'] or len(rows)!=len(raw)
            or selected['target_labels_used'] is not False):raise ValueError('bound parity panel differs')
    for row,value in zip(rows,raw,strict=True):
        r=value['request']
        if (set(row)!={'id','audio_sha256','request_sha256','rendered_sha256'}
                or set(value)!={'id','request'} or value['id']!=row['id']
                or set(r)!={'audio','questions','requirements','history','version'}):
            raise ValueError('only label-free fixed panel requests allowed')
        requests.append(ContextDecisionRequest(r['audio'],tuple(ApplicabilityQuestion(**q) for q in r['questions']),
            tuple(r['requirements']),tuple(r['history']),version=r['version']))
    population,independent=normalize(rows,receipts,requests,selected['candidate_ids'],
        expected_count=m['recordings'],decisions=WIDTHS[m['dataset']])
    if population!=selected['assets'] or independent!=selected['input_receipts']:
        raise ValueError('portable requests/cache differ from independent panel')
    return dict(dataset=m['dataset'],population=population,input_receipts=independent,
        candidate_ids=selected['candidate_ids'],requests={r['id']:q for r,q in zip(rows,requests,strict=True)},
        cache_receipts=receipts,source_rows=rows,processor_sha256=m['processor_sha256'],
        cache_root=c['cache_root'],selected_panel=selected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(prepare(c,provenance),indent=2))
