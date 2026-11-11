"""Fixed terminal-policy evaluation on all four transfer-study cells."""
import json
from pathlib import Path
from .baseline import file_hash
from .span_rule_metrics import score_rules
from .span_source_statistics import paired_source_interval

POPULATION='transfer-full-evaluation-v1'


def evaluation_rows(config):
    path=config['evaluation_protocol'];study_path=config['transfer_protocol']
    for p in (path,study_path):
        if config.get('pins',{}).get(p)!=file_hash(p):raise ValueError('pinned evaluation and study required')
    protocol=json.loads(Path(path).read_text());study=json.loads(Path(study_path).read_text())
    role=config['evaluation_role']
    if role not in ('learning','selection'):raise ValueError('explicit evaluation role required')
    release=Path(study[role+'_release'])
    if (protocol['schema']!='jev-transfer-frozen-evaluation-v1'
            or protocol['study_sha256']!=file_hash(study_path)
            or protocol['optimizer_updates']!=0 or protocol['training_admitted'] is not False
            or config['base_id']+':'+config['adapter_id'] not in protocol['policy_ids']
            or str(release)!=config['release'] or config['population']!=POPULATION):
        raise ValueError('fixed policy/role evaluation contract differs')
    if study['pins'].get(str(release/'MANIFEST.json'))!=file_hash(release/'MANIFEST.json'):
        raise ValueError('study dataset binding changed')
    manifest=json.loads((release/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        if file_hash(release/name)!=digest:raise ValueError('evaluation dataset changed')
    rows=[json.loads(s) for s in (release/(role+'.jsonl')).read_text().splitlines()]
    ids=[r['id'] for r in rows]
    source='learning_source' if role=='learning' else 'reserved_source'
    roster=[uid for key,cell in study['evaluation_cells'].items() if key.startswith(source+':') for uid in cell['request_ids']]
    if (len(ids)!=len(set(ids)) or len(roster)!=len(ids) or set(roster)!=set(ids)
            or config['request_ids']!=ids or any(r['split']!=role for r in rows)
            or (role=='selection' and any(r.get('optimization_allowed') is not False for r in rows))):
        raise ValueError('evaluation coverage or source role differs')
    return rows


def cell_metrics(study,rows,before,after):
    by_id={r['id']:r for r in rows};a={r['id']:r for r in before};b={r['id']:r for r in after}
    roster=[uid for c in study['evaluation_cells'].values() for uid in c['request_ids']]
    if (len(by_id)!=len(rows) or len(a)!=len(before) or len(b)!=len(after)
            or set(roster)!=by_id.keys() or by_id.keys()!=a.keys() or a.keys()!=b.keys()):
        raise ValueError('all four complete paired cells required')
    result={}
    for name,cell in study['evaluation_cells'].items():
        selected=[by_id[k] for k in cell['request_ids']];old=[a[r['id']] for r in selected];new=[b[r['id']] for r in selected]
        result[name]=dict(before=score_rules(selected,old),after=score_rules(selected,new),
            paired_source_intervals=paired_source_interval(selected,old,new))
        for version in ('before','after'):
            result[name][version]['scope']='Prespecified transfer cell; reserved-source role is development selection, not official TEST or cross-dataset evidence'
    return result
