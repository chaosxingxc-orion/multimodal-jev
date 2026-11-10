#!/usr/bin/env python3
"""Audit SFC-to-Jev semantic coverage before defining derived task targets."""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_consumption import restore_record
from multimodal_jev.sfc_definitions import definitions


def audit(release):
    manifest=json.loads((release/'MANIFEST.json').read_text());reports={}
    for split,pin in manifest['splits'].items():
        counts=Counter();argument_types=Counter();declared_types=Counter();call_counts=Counter();candidate_counts=Counter()
        exceptions=[];groups={}
        def anomaly(uid,kind,detail):
            counts[kind]+=1
            if len(exceptions)<30:exceptions.append(dict(id=uid,kind=kind,detail=detail))
        for kind in ('requests','targets'):
            if file_hash(release/f'{split}.{kind}.jsonl')!=pin[kind+'_sha256']:raise ValueError('native release changed')
        with (release/f'{split}.requests.jsonl').open() as ri,(release/f'{split}.targets.jsonl').open() as ti:
            for rline,tline in zip(ri,ti,strict=True):
                request,target=json.loads(rline),json.loads(tline)
                row=restore_record(request,target);uid=request['id'];counts['records']+=1
                parsed=definitions(request['input']);funcs={f['name']:f for f in parsed['functions']}
                groups.setdefault(request['source_group'],{})[target['metadata']['stage']]=parsed
                calls=row['supervision']['calls'];call_counts[len(calls)]+=1;candidate_counts[len(funcs)]+=1
                names=Counter(c['name'] for c in calls)
                counts['repeated_function_records']+=any(n>1 for n in names.values())
                counts['identical_duplicate_call_records']+=len({json.dumps(c,sort_keys=True) for c in calls})<len(calls)
                counts['calls']+=len(calls);counts['function_presence_queries']+=len(funcs)
                counts['positive_function_queries']+=sum(n in funcs for n in names)
                for f in funcs.values():
                    counts['declared_parameter_queries']+=len(f['parameters']['properties'])
                    for p in f['parameters']['properties'].values():declared_types[str(p.get('type'))]+=1
                for call in calls:
                    name=call['name'];args=call['arguments']
                    if name not in funcs:
                        anomaly(uid,'gold_function_not_in_input',name);continue
                    params=funcs[name]['parameters'];props=params['properties']
                    for key in set(params.get('required',[]))-set(args):anomaly(uid,'required_parameter_omitted',name+'.'+key)
                    for key,value in args.items():
                        argument_types[type(value).__name__]+=1;counts['arguments']+=1
                        if key not in props:
                            anomaly(uid,'gold_parameter_not_in_input',name+'.'+key);continue
                        if value=='NAN':
                            counts['literal_NAN']+=1;continue
                        if value is None:counts['literal_null']+=1
                        kind=props[key].get('type')
                        compatible={'string':type(value) is str,'array':type(value) is list,
                            'number':type(value) in (int,float),'integer':type(value) is int,
                            'boolean':type(value) is bool,'object':type(value) is dict}.get(kind)
                        if compatible is False:anomaly(uid,'parameter_type_mismatch',name+'.'+key+':'+str(kind)+'/'+type(value).__name__)
                        if 'enum' in props[key] and value not in props[key]['enum']:
                            anomaly(uid,'parameter_enum_mismatch',name+'.'+key)
                counts['calls_with_multiple_arguments']+=sum(len(c['arguments'])>1 for c in calls)
        counts['source_groups']=len(groups)
        for stages in groups.values():
            if 2 not in stages:continue
            a,b=stages[1],stages[2]
            aa={f['name']:f for f in a['functions']};bb={f['name']:f for f in b['functions']}
            counts['paired_conversations']+=1
            counts['function_order_changed']+=[f['name'] for f in a['functions']]!=[f['name'] for f in b['functions']]
            counts['function_definitions_changed']+=aa!=bb
            counts['environment_changed']+=a['environment']!=b['environment']
            counts['instruction_prefix_changed']+=a['instruction_prefix']!=b['instruction_prefix']
        if counts['records']!=pin['requests'] or counts['source_groups']!=pin['source_groups']:raise ValueError('full split coverage required')
        reports[split]=dict(counts=dict(counts),argument_value_types=dict(argument_types),declared_parameter_types=dict(declared_types),
            calls_per_record=dict(call_counts),functions_per_input=dict(candidate_counts),first_30_schema_anomalies=exceptions)
    return dict(schema='sfc-jev-definition-audit-v1',release=str(release.resolve()),manifest_sha256=file_hash(release/'MANIFEST.json'),
        parser_sha256=file_hash(Path(__file__).parents[2]/'src/multimodal_jev/sfc_definitions.py'),splits=reports,
        scope='Complete data-definition audit only. Inputs parsed independently of labels; labels used solely for conversion coverage diagnostics, not model or prompt selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.release)
    with a.output.open('xb') as f:f.write(canonical(report))
    print(json.dumps({s:v['counts'] for s,v in report['splits'].items()},indent=2))
