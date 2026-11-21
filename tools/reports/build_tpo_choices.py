"""Bind a gold-blind candidate proposal to the prior authenticated TRAIN view."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from multimodal_jev.context_choice import shortlist, lengths, varied_slate
from multimodal_jev.context_reference_dataset import validate_record
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.build_tpo_view import read, verified
from multimodal_jev.baseline import file_hash


def build(c):
    src, ref, dst = map(Path,(c['reference_view'],c['reference'],c['output']))
    check_paths({'target':dst})
    m = read(verified(src/'MANIFEST.json', c['reference_view_sha256']))
    r = read(verified(ref/'MANIFEST.json', m['reference_sha256']))
    spec = c['choice']
    variable = 'candidate_lengths' in spec
    required_modes = {'open','noul'} if variable else {'open','closed','noul'}
    if (m['schema'] != 'jev-tpo-reward-view-v1' or r['identity']['policy_id'] != m['source_policy']
            or r['identity']['candidate_ids'] != m['candidate_ids'] or set(spec['cycle']) != required_modes):
        raise ValueError('incompatible prior or task cycle')
    for n,h in m['files'].items(): verified(src/n,h)
    records = read(verified(ref/'records.json',r['files']['records.json']))
    rows = read(src/'rows.json')
    for uid,row in rows.items():
        rec = records[row['request_key']]
        validate_record(rec,r['identity'])
        if row['request_key'] != rec['request_id']: raise ValueError('reference key mismatch')
        if not variable: row['slate'] = shortlist(rec['prob'], m['candidate_ids'], uid, spec)
    batches = read(src/'batches.json')
    counts = Counter()
    widths = lengths(spec,len(m['candidate_ids'])) if variable else []
    group_cursor = 0
    for idx,batch in enumerate(batches):
        mode = spec['cycle'][idx%len(spec['cycle'])]
        batch['mode'] = mode
        micro = m['schedule']['microbatch']
        for offset,slot in enumerate(batch['slots']):
            if variable and mode=='open':
                uid=slot['id']
                scores=records[rows[uid]['request_key']]['prob']
                k=widths[(group_cursor + offset//micro) % len(widths)]
                slot['slate'],slot['proposal']=varied_slate(scores,m['candidate_ids'],uid,spec,k,batch['epoch'])

            if slot['padding']: continue
            row = rows[slot['id']]
            counts[mode if mode!='open' else ('open_hit' if row['gold'] in (slot['slate'] if variable else row['slate']) else 'open_miss')] += 1
        if variable and mode=='open': group_cursor += len(batch['slots'])//micro
    if set(counts) != (required_modes-{'open'})|{'open_hit','open_miss'}: raise ValueError('missing task support')
    if variable:
        from tools.reports.audit_choice_coverage import summarize
        audit=summarize(rows,batches,m['candidate_ids'])
        if (set(cell['k'] for cell in audit['cells'].values()) != set(widths) or
                not audit['recordings_with_both_none_targets'] or
                any(not cell['hits'] or (cell['k'] < len(m['candidate_ids']) and not cell['misses'])
                    for cell in audit['cells'].values())):
            raise ValueError('missing variable-width coverage supervision')
    dst.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(src/'targets.pt',dst/'targets.pt')
    commit_json(dst/'rows.json',rows); commit_json(dst/'batches.json',batches)
    commit_json(dst/'MANIFEST.json',dict(m, schema='jev-tpo-choices-v1', choice=spec, task_counts=dict(counts),
        prior_manifest=c['reference_view_sha256'],**({'sampling_audit':audit} if variable else {}),files={n:file_hash(dst/n) for n in m['files']}))
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('local');a=p.parse_args()
    print(json.dumps(build(load_config(a.config,a.local)[0])))
