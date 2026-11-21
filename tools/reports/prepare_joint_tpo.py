"""Build the paired joint TPO view from an authenticated Choice view."""
from collections import Counter
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_choice import lengths, varied_slate
from multimodal_jev.context_reference_dataset import validate_record
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.build_tpo_view import read, verified

def prepare(joint, entry, data, parent_path, parent, repo):
    plan=read(repo/joint['design']); lock=read(repo/plan['comparison'])
    m=read(verified(data/'MANIFEST.json',lock['binding']['target_sha']))
    for name,sha in m['files'].items(): verified(data/name,sha)
    if (file_hash(parent_path)!=lock['binding']['parent_sha'] or m['source_policy']!=plan['source_policy'] or
        parent['source_policy_id']!=plan['source_policy'] or m['schema']!='jev-tpo-choices-v1' or
        m['updates']!=plan['updates'] or len(parent['physical_devices'])!=lock['binding']['world_size'] or
        m['view_sha256']!=parent['training_dataset']['manifest_sha256'] or
        m['reference_sha256']!=parent['reference_dataset']['manifest_sha256'] or
        any(m['schedule'][k]!=plan[k] for k in ('epochs','seed','global_batch','microbatch'))):
        raise ValueError('joint plan binding differs')
    if len(joint['inference_devices'])!=parent['batch_config']['inference_instances'] or set(parent['physical_devices'])&set(joint['inference_devices']):
        raise ValueError('TP1 inference services must use separate devices')
    native=Path(joint['native_reference']); nm=read(verified(native/'MANIFEST.json',m['reference_sha256']))
    if nm['identity']['policy_id']!=m['source_policy'] or nm['identity']['candidate_ids']!=m['candidate_ids']: raise ValueError('reference policy differs')
    records=read(verified(native/'records.json',nm['files']['records.json'])); population=read(verified(native/'population.json',nm['files']['population.json']))
    rows=read(data/'rows.json'); schedule=read(data/'batches.json'); ids=m['candidate_ids']; probs={}
    if len(rows)!=plan['recordings'] or len(schedule)!=plan['updates']:
        raise ValueError('wrong experiment population')
    for uid,row in rows.items():
        if population[uid]!=row['request_key'] or not 0<=row['gold']<len(ids): raise ValueError('input/gold differs')
        rec=records[row['request_key']]; validate_record(rec,nm['identity']); probs[uid]=rec['prob']
    widths=lengths(m['choice'],len(ids)); micro=plan['microbatch']; cursor=0; exposure=Counter(); counts=Counter(); classes=Counter()
    for batch in schedule:
        old=batch['mode']; batch['mode']='joint'; batch['prior_mode']=old
        if len(batch['slots'])!=plan['global_batch'] or not 0<=batch['epoch']<plan['epochs']:
            raise ValueError('unregistered batch geometry')
        for pos,slot in enumerate(batch['slots']):
            uid=slot['id']
            if old=='noul':
                k=widths[(cursor+pos//micro)%len(widths)]; slot['slate'],slot['proposal']=varied_slate(probs[uid],ids,uid,m['choice'],k,batch['epoch'])
            if old not in ('open','noul'): raise ValueError('unregistered view')
            slate=slot['slate']
            if (not slate or len(set(slate))!=len(slate) or any(type(i) is not int or not 0<=i<len(ids) for i in slate)
                    or len(slate)!=len(batch['slots'][pos-pos%micro]['slate'])
                    or type(slot['padding']) is not bool or slot['weight']!=int(not slot['padding'])):
                raise ValueError('invalid slate geometry or padding')
            if not slot['padding']:
                exposure[batch['epoch'],uid]+=1; counts['choice_hit' if rows[uid]['gold'] in slot['slate'] else 'choice_miss']+=1
                classes[ids[rows[uid]['gold']]]+=1
                counts['retained_choice' if old=='open' else 'added_choice']+=1
        if old=='noul': cursor+=len(batch['slots'])//micro
    if exposure!=Counter({(e,u):1 for e in range(plan['epochs']) for u in rows}) or set(classes)!=set(ids) or not counts['choice_hit'] or not counts['choice_miss']: raise ValueError('incomplete joint exposure')
    receipt=read(Path(joint['control_receipt']))
    if receipt['target_sha']!=lock['binding']['target_sha'] or receipt['parent_sha']!=lock['binding']['parent_sha'] or receipt['world_size']!=lock['binding']['world_size'] or receipt['window']!='reference' or receipt['passed'] is not True: raise ValueError('reference control not admitted')
    dest=entry/'targets'; dest.mkdir()
    for name,payload in [('rows.json',rows),('batches.json',schedule),('native-probs.json',probs)]: commit_json(dest/name,payload)
    commit_json(dest/'MANIFEST.json',dict(schema='jev-tpo-joint-view-v1',source_policy=m['source_policy'],release_sha256=m['release_sha256'],reference_sha256=m['reference_sha256'],candidate_ids=ids,updates=plan['updates'],schedule=m['schedule'],parent_view_sha256=lock['binding']['target_sha'],counts=dict(counts),per_class_exposures=dict(classes),real_exposures=sum(exposure.values()),files={p.name:file_hash(p) for p in sorted(dest.iterdir())}))
    return dict(plan,target_sha=file_hash(dest/'MANIFEST.json'),parent_sha=file_hash(parent_path),
                old_choice_anchor=receipt,training_devices=parent['physical_devices'],inference_devices=joint['inference_devices'])
