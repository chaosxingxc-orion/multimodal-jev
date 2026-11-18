#!/usr/bin/env python3
"""Compare complete CPU processor probes; never admit training or NPU parity."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def key(row):
    return row['audio_id'], row['panel_id'], row['arm']


IDENTITY = ('audio_sha256', 'audio_samples', 'representation', 'native_layout',
            'template_sha256', 'expanded_contract_sha256')
OBSERVATIONS = ('tokens', 'audio_tokens', 'positions', 'mappings', 'input_ids_sha256')


def compare_records(before, after):
    a = {key(r): r for r in before}
    b = {key(r): r for r in after}
    require(len(a)==len(before) and len(b)==len(after), 'duplicate request coordinate')
    require(bool(a) and a.keys()==b.keys(), 'complete equal populations required')
    differences = []
    for k in sorted(a):
        require(all(a[k][f]==b[k][f] for f in IDENTITY), 'input identity changed')
        changed = {f:dict(before=a[k][f], after=b[k][f]) for f in OBSERVATIONS if a[k][f]!=b[k][f]}
        if changed:
            differences.append(dict(coordinate=list(k), fields=changed))
    return dict(requests=len(a), identical_requests=len(a)-len(differences),
                all_recorded_inputs_equal=not differences, differences=differences)


def load_run(root):
    config=read(root/'config.json'); result=read(root/'run/result.json')
    terminal=read(root/'terminal.json'); state=terminal['container_state']
    require(state['Status']=='exited' and not state['Running'] and state['ExitCode']==0
            and not state['OOMKilled'], 'successful terminal required')
    require(terminal['container_image']==config['image']==result['image'], 'actual image differs')
    require(sha(root/'config.json')==result['config_sha256'], 'changed config')
    require(sha(root/'run/records.jsonl')==result['records_sha256'], 'changed records')
    for p,h in config['pins'].items():
        require(sha(p)==h, 'changed dependency '+p)
    for asset in config['audio']:
        require(sha(asset['audio'])==asset['sha256'], 'audio file changed')
    implementation=root/'source/tools/jev/probe_semantic_actor_context.py'
    require(sha(implementation)==result['implementation_sha256'], 'changed implementation')
    require(result['context_accepted'] is False and result['training_dispatch_allowed'] is False,
            'diagnostic scope required')
    records=[json.loads(line) for line in (root/'run/records.jsonl').read_text().splitlines()]
    templates=[json.loads(line) for line in Path(config['templates']).read_text().splitlines()]
    expected={(a['id'],t['panel_id'],t['arm']) for a in config['audio'] for t in templates}
    require(len(records)==result['requests']==len(expected) and {key(r) for r in records}==expected,
            'incomplete Cartesian request coverage')
    template_map={(t['panel_id'],t['arm']):t for t in templates}
    audio_map={a['id']:a for a in config['audio']}
    for r in records:
        t=template_map[r['panel_id'],r['arm']]
        require(r['audio_sha256']==audio_map[r['audio_id']]['sha256'], 'audio binding differs')
        require(all(r[f]==t[f] for f in IDENTITY[2:]), 'template binding differs')
        require(r['overflow']=={str(n):r['tokens']>n for n in (8192,16384,32768,65536)}, 'overflow differs')
    for arm,summary in result['per_arm'].items():
        rr=[r for r in records if r['arm']==arm]
        require(summary==dict(requests=len(rr),min_tokens=min(r['tokens'] for r in rr),
            max_tokens=max(r['tokens'] for r in rr),overflow={str(n):sum(r['tokens']>n for r in rr)
            for n in (8192,16384,32768,65536)}), 'summary differs')
    return config,result,records


def compare_runs(before, after):
    a,ar,aa=load_run(before);b,br,bb=load_run(after)
    launch=read(after/'launch.json');terminal=read(after/'terminal.json')
    require(launch['config_sha256']==sha(after/'config.json'), 'launched config differs')
    inspection_path=after/'container-inspection.json'
    require(terminal['inspection_sha256']==sha(inspection_path), 'inspection evidence changed')
    inspection=read(inspection_path)
    require(isinstance(inspection,list) and len(inspection)==1, 'one container inspection required')
    actual=inspection[0]
    require(actual['Id']==launch['container_id']==terminal['container_id'], 'container identity differs')
    require(actual['Image']==terminal['container_image']==b['image']
            and actual['State']==terminal['container_state'], 'inspection differs from terminal')
    require(actual['HostConfig']['NetworkMode']=='none' and not actual['HostConfig']['Devices'],
            'isolated CPU diagnostic required')
    argv=launch['argv'];command=argv[argv.index(b['image'])+1:]
    require(actual['Config']['Entrypoint']==['timeout'] and actual['Config']['Cmd']==command,
            'executed command differs')
    require(a['audio']==b['audio'] and a['model']==b['model'], 'assets or model changed')
    require(Path(a['templates']).read_bytes()==Path(b['templates']).read_bytes(), 'templates changed')
    require(b['image']==b['fixed_training_image'] and br['fixed_training_image_used'] is True,
            'comparison must use actual fixed training image')
    report=compare_records(aa,bb)
    report.update(schema='semantic-fixed-image-context-comparison-v1',
        before_root=str(before),after_root=str(after),before_result_sha256=sha(before/'run/result.json'),
        after_result_sha256=sha(after/'run/result.json'),after_terminal_sha256=sha(after/'terminal.json'),
        after_launch_sha256=sha(after/'launch.json'),after_inspection_sha256=sha(inspection_path),
        implementation_sha256=sha(__file__),before_image=a['image'],after_image=b['image'],
        before_versions=ar['versions'],after_versions=br['versions'],
        processor_source_equal=ar['processor_source_sha256']==br['processor_source_sha256'],
        per_arm=br['per_arm'],context_accepted=False,training_dispatch_allowed=False,
        scope='Complete paired CPU processor records in two identified images. Token ID equality uses SHA256, not retained raw IDs. No model forward, feature tensor equality, full population, NPU resources or policy probability acceptance.')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();require(not a.output.exists(),'output exists')
    report=compare_runs(a.before,a.after)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='differences'},indent=2))
