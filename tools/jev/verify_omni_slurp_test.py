#!/usr/bin/env python3
"""Independently accept and rescore a complete frozen native SLURP TEST."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_slurp_test_contract import select_native_test
from score_slurp_native import score_run
from watch_qwen3_omni_probe import inspect as inspect_container


def compare_values(original, rescored, path=''):
    """Integer counts are exact; tolerate only rounding in official float sums."""
    if type(original) is not type(rescored):
        raise ValueError('score type differs: '+path)
    if isinstance(original, dict):
        if original.keys()!=rescored.keys():raise ValueError('score keys differ: '+path)
        return [d for k in original for d in compare_values(original[k],rescored[k],path+'/'+k)]
    if isinstance(original, list):
        if len(original)!=len(rescored):raise ValueError('score length differs: '+path)
        return [d for i,(a,b) in enumerate(zip(original,rescored)) for d in compare_values(a,b,path+'/'+str(i))]
    if isinstance(original, float):
        if not (math.isfinite(original) and math.isfinite(rescored)
                and math.isclose(original,rescored,rel_tol=1e-12,abs_tol=1e-12)):
            raise ValueError('official score differs: '+path)
        return [] if original==rescored else [dict(path=path,original=original,
            independent=rescored,absolute_difference=abs(original-rescored))]
    if original!=rescored:raise ValueError('score value differs: '+path)
    return []


def validate_records(rows, predictions, result, cfg):
    expected={r['id']:r for r in rows}
    if (len(expected)!=len(rows) or len(predictions)!=len(rows)
            or len({p['example_id'] for p in predictions})!=len(rows)
            or {p['example_id'] for p in predictions}!=set(expected)
            or result['requests']!=len(rows) or result['predictions']!=len(rows)):
        raise ValueError('complete unique native TEST predictions required')
    for p in predictions:
        row=expected[p['example_id']]
        if (type(p['replicate']) is not int or p['replicate']!=0 or p['split']!='test'
                or p['audio_condition']!='actual' or p['source_group']!=row['source_group']
                or p['request_sha256']!=hashlib.sha256(canonical(row)).hexdigest()):
            raise ValueError('prediction request binding differs')
        if (type(p['input_tokens']) is not int or p['input_tokens']<=0
                or p['input_tokens']+cfg['max_new_tokens']>cfg['max_context_tokens']
                or len(p['token_ids'])>cfg['max_new_tokens']):
            raise ValueError('prediction exceeds registered context or output budget')
        if (type(p['length_limited']) is not bool
                or p['length_limited']!=(p['finish_reason']=='length')
                or type(p['seconds']) not in (int,float)
                or not math.isfinite(p['seconds']) or p['seconds']<0):
            raise ValueError('invalid generation accounting')
    if result['length_limited']!=sum(p['length_limited'] for p in predictions):
        raise ValueError('truncation accounting differs')


def verify(root, launch_record, native_probe, output):
    if output.exists():raise FileExistsError('preserve prior independent acceptance')
    # Fail before reading TEST predictions unless both application and supervisor ended.
    for path in (root/'evaluation-terminal.json',root/'run/terminal.json'):
        if json.loads(path.read_text())['exit_code']!=0:
            raise ValueError('successful supervisor and application required')
    evidence=json.loads(launch_record.read_text())
    pinned=[j for j in evidence['jobs'] if Path(j['root']).resolve()==root.resolve()]
    if len(pinned)!=1:raise ValueError('unique prior native TEST launch required')
    pinned=pinned[0];launch=json.loads((root/'launch.json').read_text())
    for key in ('container_id','image','physical_devices','config_sha256'):
        if launch[key]!=pinned[key]:raise ValueError('registered launch changed: '+key)
    if launch['physical_devices']!=[6,7] or launch['source_commit']!=pinned['supervisor']['source_commit']:
        raise ValueError('registered source and devices required')
    item=inspect_container(launch['container_id'])
    terminal=json.loads((root/'docker-terminal.json').read_text())
    if (item['Id']!=launch['container_id'] or item['Image']!=launch['image']
            or item['State']['Status']!='exited' or item['State']['ExitCode']!=0
            or terminal['container_id']!=item['Id'] or terminal['state']['Status']!='exited'
            or terminal['state']['ExitCode']!=0):
        raise ValueError('exact successful native TEST container required')
    run=root/'run';identity=json.loads((run/'identity.json').read_text());cfg=identity['config']
    result=json.loads((run/'result.json').read_text())
    if (result['status']!='passed' or result['weight_updates']!=0
            or result['model_files_unchanged'] is not True or result['audio_condition']!='actual'
            or file_hash(root/'config.json')!=launch['config_sha256']
            or cfg!=json.loads((root/'config.json').read_text())
            or file_hash(run/'predictions.jsonl')!=result['predictions_sha256']):
        raise ValueError('frozen TEST artifacts changed or incomplete')
    for field,key in [('instruction','instruction_sha256'),('requests','requests_sha256'),('asset_audit','asset_audit_sha256')]:
        if file_hash(cfg[field])!=identity[key]:raise ValueError('input identity changed: '+field)
    sources={**{'src/multimodal_jev/'+n:d for n,d in identity['source_sha256'].items()},
        'tools/jev/infer_qwen3_omni_probe.py':identity['runner_sha256'],
        'src/multimodal_jev/omni_slurp_test_contract.py':identity['population_selector_sha256']}
    for name,digest in sources.items():
        committed=subprocess.check_output(['git','show',launch['source_commit']+':'+name])
        if hashlib.sha256(committed).hexdigest()!=digest or file_hash(root/'source'/name)!=digest:
            raise ValueError('frozen inference source differs: '+name)
    prior=json.loads((native_probe/'run/identity.json').read_text())
    if json.loads((native_probe/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('successful prior TRAIN probe required')
    for key in ('versions','model_files','instruction_sha256','taxonomy_sha256'):
        if identity[key]!=prior[key]:raise ValueError('TRAIN baseline identity differs: '+key)
    for key in ('audio_model','dtype','seed','sampling','max_context_tokens','max_new_tokens','route','task_format','model_family'):
        if cfg[key]!=prior['config'][key]:raise ValueError('TRAIN baseline config differs: '+key)
    rows=select_native_test(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    if json.loads((run/'selection.json').read_text())['ids']!=[r['id'] for r in rows]:
        raise ValueError('native TEST selection order differs')
    predictions=[json.loads(line) for line in (run/'predictions.jsonl').open()]
    validate_records(rows,predictions,result,cfg)
    # Reserve once before any scoring output; failures remain inspectable, never replayed.
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'intent.json',dict(root=str(root.resolve()),launch_record_sha256=file_hash(launch_record),
        identity_sha256=file_hash(run/'identity.json'),predictions_sha256=result['predictions_sha256']))
    rescored=score_run(run,Path(cfg['requests']).with_name('test.targets.jsonl'),
        Path(launch['scorer_repository']),output/'score')
    original=json.loads((root/'score/result.json').read_text())
    differences=compare_values(original,rescored)
    if file_hash(root/'score/official-predictions.jsonl')!=file_hash(output/'score/official-predictions.jsonl'):
        raise ValueError('independent normalized predictions differ')
    if (rescored['examples']!=13078 or rescored['full_official_split'] is not True
            or rescored['format_valid']!=result['format_valid'] or rescored['schema_valid']!=result['schema_valid']):
        raise ValueError('independent full coverage or format counts differ')
    report=dict(schema='jev-native-slurp-test-acceptance-v1',root=str(root.resolve()),
        container_id=item['Id'],source_commit=launch['source_commit'],examples=len(rows),
        source_groups=len({r['source_group'] for r in rows}),weight_updates=0,
        normalized_predictions_exact=True,score_float_differences=differences,
        relative_tolerance=1e-12,absolute_tolerance=1e-12,
        original_score_sha256=file_hash(root/'score/result.json'),independent_score_sha256=file_hash(output/'score/result.json'),
        length_limited=result['length_limited'],metrics=rescored['metrics'],
        scope='Complete frozen native SLURP TEST acceptance and independent rescore; no new inference or optimization')
    atomic_json(output/'acceptance.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('root','launch-record','native-probe','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.launch_record,args.native_probe,args.output),indent=2))
