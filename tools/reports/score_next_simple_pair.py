"""Compare two collected HF cells from their pinned independent scoring receipts."""
import argparse
import json
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.paired_event_decisions import compare
from tools.reports.jev_train_confusions import pinned


def score(c):
    design=json.loads(pinned(Path(c['design']['path']),c['design']['sha256']))
    def cell(ref):
        root=Path(ref['root']);arm=ref['arm'];seal=Path(ref['seal'])
        audit=json.loads(pinned(root/'paired-score.json',ref['score_sha256']))
        term=json.loads((root/'terminal.json').read_bytes())
        if term!=dict(exit_code=0,score_sha256=ref['score_sha256']):raise ValueError('successful score required')
        m=json.loads(pinned(seal/'MANIFEST.json',audit['seal_sha256']))
        get=lambda name:json.loads(pinned(seal/name,m['files'][name]))
        policy=audit['policy_id'];proof=audit['arms'][arm]
        pred=json.loads(pinned(Path(ref['records'])/f'hf-{arm}-collected.json',proof['predictions_file_sha256']))
        if policy!=ref['policy_id'] or pred['policy_id']!=policy or pred['panel_sha256']!=proof['fingerprint']:
            raise ValueError('bound endpoint identity required')
        canonical=get('inference/native/panel.json')
        panel=get(f'inference/{arm}/panel.json');ids=panel['candidate_ids']
        targets=get('scoring/targets.json');rows=get(f'inference/{arm}/rows.json')
        data={}
        for p,t,r in zip(pred['records'],targets,rows,strict=True):
            if p['id']!=t['id'] or r['id']!=t['id'] or t['audio_sha256']!=r['audio_sha256'] or p['id'] in data:
                raise ValueError('paired recording identity differs')
            data[p['id']]=dict(zip(ids,p['prob'],strict=True))
        return m,canonical,panel,targets,data
    a,b=c['left'],c['right']
    l,base,x,targets,p=cell(a);r,native,y,truth,q=cell(b)
    if (base!=native or targets!=truth or l['design_sha256']!=c['design']['sha256']
            or r['design_sha256']!=l['design_sha256'] or l['parent_sha256']!=r['parent_sha256']):
        raise ValueError('paired task, design and population required')
    if c['comparison']=='same_policy_layout':
        if a['policy_id']!=b['policy_id'] or x['candidate_ids']==y['candidate_ids']:raise ValueError('one policy and distinct layouts required')
    elif c['comparison']=='same_layout_policy':
        if a['policy_id']==b['policy_id'] or x!=y:raise ValueError('same layout and distinct policies required')
    else:raise ValueError('unknown comparison')
    collect = l.get('diagnostic') == 'full_train_unseen_layout'
    if collect and (r!=l or c['comparison']!='same_policy_layout'
                    or (a['arm'],b['arm'])!=('native','unseen')
                    or a['policy_id']!=c['expected_policy_id']):
        raise ValueError('full TRAIN collection requires the frozen Stage1 policy and both arms of one seal')
    return dict(comparison=c['comparison'],left_policy=a['policy_id'],right_policy=b['policy_id'],
                left_arm=a['arm'],right_arm=b['arm'],scope='source TRAIN mechanism diagnostic only',
                **compare(base['candidate_ids'],targets,p,q,interval=design['interval'],cluster=c['cluster'],
                          collect_ids=collect))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);commit_json(c['output'],score(c))
