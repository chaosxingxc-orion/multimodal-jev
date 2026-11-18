"""Audit SFC decision-task structure without consuming held-out target labels.

Input schema is inspected on all splits; label projection diagnostics read only
TRAIN. No semantic completeness, audio decoding or model efficacy certification.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.sfc_consumption import SPLITS,validate_request,restore_record
from multimodal_jev.sfc_definitions import definitions
from multimodal_jev.sfc import source_identity


def audit(config):
    root=Path(config['release']);manifest_path=root/'MANIFEST.json'
    if file_hash(manifest_path)!=config['manifest_sha256']:
        raise ValueError('pinned SFC native manifest required')
    manifest=json.loads(manifest_path.read_text())
    if manifest['identity']['dataset']!='sfc-native' or set(manifest['splits'])!=set(SPLITS):
        raise ValueError('complete native SFC release required')
    reports={};train_rows=[];input_hashes={};audio_owners={};function_sets={}
    for split in SPLITS:
        path=root/f'{split}.requests.jsonl';digest=file_hash(path);pin=manifest['splits'][split]
        if digest!=pin['requests_sha256']:raise ValueError('native request file changed')
        input_hashes[split]=digest;ids=set();groups=set();levels=Counter();widths=Counter();audios=Counter()
        functions=set();environment=0;history=0;empty_history=0;conversations={}
        with path.open() as stream:
            for line in stream:
                row=json.loads(line);validate_request(row)
                if row['id'] in ids or row['split']!=split:raise ValueError('duplicate ID or wrong split')
                ids.add(row['id']);groups.add(row['source_group'])
                identity=source_identity([row['id'].removeprefix('sfc:')])
                if identity['source_group']!=row['source_group'] or identity['official_split']!=split:
                    raise ValueError('source group or split differs')
                if identity['stage']!=len(row['input']['audios']):raise ValueError('audio stage differs')
                conversation=conversations.setdefault(row['source_group'],{})
                if identity['stage'] in conversation:raise ValueError('duplicate conversation stage')
                conversation[identity['stage']]=row
                parsed=definitions(row['input']);names=[f['name'] for f in parsed['functions']]
                functions.update(names);levels[identity['level']]+=1;widths[len(names)]+=1
                audios[len(row['input']['audios'])]+=1;environment+=bool(parsed['environment'])
                history+=len(row['input']['messages'])==3
                empty_history+=len(row['input']['messages'])==3 and not row['input']['messages'][1]['content'].strip()
                for h in row['audio_sha256']:
                    if h in audio_owners and audio_owners[h]!=split:raise ValueError('audio hash crosses splits')
                    audio_owners[h]=split
                if split=='train_id':train_rows.append(row)
        if len(ids)!=pin['requests'] or len(groups)!=pin['source_groups']:
            raise ValueError('complete input population required')
        for group,stages in conversations.items():
            if set(stages)!=({1,2} if group.startswith('level_3') else {1}):
                raise ValueError('incomplete supplied conversation')
            if 2 in stages and (stages[1]['input']['audios']!=stages[2]['input']['audios'][:1]
                    or stages[1]['audio_sha256']!=stages[2]['audio_sha256'][:1]):
                raise ValueError('ordered conversation audio prefix differs')
        function_sets[split]=functions
        reports[split]=dict(recordings=len(ids),source_groups=len(groups),levels=dict(levels),
            candidate_count_histogram=dict(sorted(widths.items())),audio_count_histogram=dict(sorted(audios.items())),
            unique_defined_functions=len(functions),nonempty_environment=environment,
            supplied_assistant_history=history,empty_assistant_history=empty_history,
            targets_opened=False)
    targets=root/'train_id.targets.jsonl';target_hash=file_hash(targets)
    if target_hash!=manifest['splits']['train_id']['targets_sha256']:raise ValueError('TRAIN targets changed')
    calls_hist=Counter();active_hist=Counter();by_level={};total_candidates=positive=0;repeated=nan_rows=0
    def contains_nan(value):
        if isinstance(value,dict):return any(contains_nan(v) for v in value.values())
        if isinstance(value,list):return any(contains_nan(v) for v in value)
        return value=='NAN'
    with targets.open() as stream:
        for row,line in zip(train_rows,stream,strict=True):
            record=restore_record(row,json.loads(line));calls=record['supervision']['calls']
            names={f['name'] for f in definitions(row['input'])['functions']}
            active={call['name'] for call in calls}
            if not active<=names:raise ValueError('TRAIN target function outside supplied definitions')
            total_candidates+=len(names);positive+=len(active)
            calls_hist[len(calls)]+=1;active_hist[len(active)]+=1
            repeated+=len(calls)>len(active);nan_rows+=any(contains_nan(call['arguments']) for call in calls)
            level=record['metadata']['level'];stats=by_level.setdefault(level,Counter())
            stats['queries']+=1;stats['multiple_active_functions']+=len(active)>1
            stats['empty_call_list']+=not calls;stats['repeated_function_calls']+=len(calls)>len(active)
    reports['train_id']['targets_opened']=True
    return dict(schema='sfc-decision-structural-admission-v1',release_manifest_sha256=config['manifest_sha256'],
        request_files_sha256=input_hashes,train_targets_sha256=target_hash,splits=reports,
        input_function_inventory=dict(train_defined=len(function_sets['train_id']),
            test_id_defined=len(function_sets['test_id']),test_ood_defined=len(function_sets['test_ood']),
            test_ood_defined_also_in_train=len(function_sets['test_ood']&function_sets['train_id']),
            test_ood_defined_not_in_train=len(function_sets['test_ood']-function_sets['train_id']),
            scope='Available candidates only, including distractors; not target function exposure'),
        train_projection=dict(call_count_histogram=dict(sorted(calls_hist.items())),
            active_function_count_histogram=dict(sorted(active_hist.items())),
            repeated_function_queries=repeated,missing_argument_NAN_queries=nan_rows,
            candidates=total_candidates,positive_memberships=positive,negative_memberships=total_candidates-positive,
            levels={k:dict(v) for k,v in by_level.items()}),
        heldout_targets_opened=False,semantic_completeness_verified=False,audio_bytes_reverified=False,
        candidate_task='membership_in_official_current_turn_called_function_set',
        applicability_claim_allowed=False,training_dispatch_allowed=False,evaluation_dispatch_allowed=False,
        pending=['native call-set projection protocol and scorer',
                 'environment and ordered multi-audio supplied-history compiler',
                 'dynamic candidate input budget and actual backend parity',
                 'historical exposure declaration and held-out scoring admission'],
        scope='Full input structure and TRAIN-only lossless label projection audit; '
              'not exhaustive semantic applicability, target accuracy or interaction success.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    result=audit(config);result['config_provenance']=provenance
    output=Path(config['output']);output.parent.mkdir(parents=True,exist_ok=True)
    commit_json(output,result)
    print(json.dumps(result,indent=2))
