"""Seal label-free MINDS inference around the existing source-selected pair.

The source pair is selected solely by SLURP selection, identical to FSC's
pair. This tool neither reads target labels nor dispatches a device workload.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from tools.jev.prepare_fsc_context_source import verify as verify_source
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.prepare_fsc_context_package import IMAGE, PREFIX, MODEL
from tools.reports.run_minds_context_pair import configs

CPU_ROOT=PREFIX+'minds-processor-cpu-v1'
CACHE=CPU_ROOT+'/run/cache'
AUDIO_SOURCE=CPU_ROOT+'/assets/audio'
AUDIO_DEST='/home/xingchao/workspace/multimodal-jev/.local/decision-objective-research/minds14-context-inputs-16k-v1/audio'


def read(p):return json.loads(Path(p).read_text())


def relocate(value,old,new):
    if isinstance(value,str):return value.replace(old,new)
    if isinstance(value,list):return [relocate(v,old,new) for v in value]
    if isinstance(value,dict):return {relocate(k,old,new):relocate(v,old,new) for k,v in value.items()}
    return value


def verify(package,digest):
    root=Path(package)
    if root.is_symlink() or sha(root/'PACKAGE.json')!=digest:raise ValueError('MINDS manifest differs')
    m=read(root/'PACKAGE.json')
    if (m['status']!='prepared_minds_pair_no_dispatch' or m['image']!=IMAGE
            or not re.fullmatch(re.escape(PREFIX)+r'minds-context-pair-[a-z0-9-]+',m['remote_root'])):
        raise ValueError('isolated fixed-image MINDS package required')
    actual={}
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('MINDS package symlink forbidden')
        if p.is_file() and p!=root/'PACKAGE.json':actual[str(p.relative_to(root))]=sha(p)
    if actual!=m['files']:raise ValueError('MINDS files differ')
    c=read(root/'config.json');remote=m['remote_root']
    if (c['model']!=MODEL or c['physical_devices']!=[7] or c['inputs']['cache_root']!=CACHE
            or c['source_package']!=remote+'/frozen-source'
            or c['source_manifest_sha256']!=m['source_manifest_sha256']):
        raise ValueError('MINDS model/device/source/cache binding differs')
    expected={remote+'/'+k:h for k,h in actual.items() if k.startswith(('source/','frozen-source/','inputs/','processor/'))}
    expected.update(m['external_pins'])
    if c['pins']!=expected:raise ValueError('MINDS dependency pins differ')
    if MODEL+'/model.safetensors' not in m['external_pins']:raise ValueError('base weights not pinned')
    for p,h in m['external_pins'].items():
        if Path(p).parent!=Path(MODEL) or sha(p)!=h:raise ValueError('base pin differs')
    configs(relocate(c,remote,str(root.resolve())))
    return m


def prepare(spec,output,remote_root):
    if not re.fullmatch(re.escape(PREFIX)+r'minds-context-pair-[a-z0-9-]+',remote_root):
        raise ValueError('isolated MINDS root required')
    source=Path(spec['source_package']);m=verify_source(source,spec['source_manifest_sha256'])
    output=Path(output).resolve();inputs=Path(spec['inputs']);processor=Path(spec['processor_receipts_root'])
    for root in (source,inputs,processor,Path(MODEL)):
        if output.is_relative_to(root.resolve()) or root.resolve().is_relative_to(output):
            raise ValueError('independent package required')
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'test_package':output})
    output.mkdir(exist_ok=False)
    shutil.copytree(source,output/'frozen-source');(output/'inputs').mkdir()
    for name in ('MANIFEST.json','inputs.jsonl','definitions.json','conversion.jsonl'):
        shutil.copyfile(inputs/name,output/'inputs'/name)
    (output/'processor').mkdir()
    for name in ('MANIFEST.json','inputs.jsonl'):
        shutil.copyfile(processor/name,output/'processor'/name)
    bindings=read(spec['input_config'])
    extras={'processor_audit':'audit.json','protocol':'protocol.json',
            'preparation_transport':'preparation-transport.json',
            'verification_transport':'verification-transport.json',
            'actual_verification':'actual-verification.json'}
    for key,name in extras.items():
        entry=bindings[key]
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Input evidence changed: '+key)
        shutil.copyfile(entry['path'],output/'processor'/name)
    shutil.copyfile(bindings['conversion_verification'],output/'processor/conversion-verification.json')
    for folder in ('src','tools/jev'):
        for p in Path(folder).rglob('*.py'):
            target=output/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    for name in ('minds_context_runtime.py','preflight_minds_context_inputs.py','prepare_minds_context_inputs.py','run_minds_context_pair.py'):
        target=output/'source/tools/reports'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path('tools/reports')/name,target)
    external={str(p):sha(p) for p in Path(MODEL).iterdir()
        if p.is_file() and (p.suffix in ('.json','.txt','.jinja') or p.name=='model.safetensors')}
    c=dict(source_package=remote_root+'/frozen-source',source_manifest_sha256=spec['source_manifest_sha256'],
        model=MODEL,physical_devices=[7],inputs=dict(inputs=remote_root+'/inputs',
            input_manifest_sha256=sha(output/'inputs/MANIFEST.json'),
            processor_manifest=dict(path=remote_root+'/processor/MANIFEST.json',sha256=sha(output/'processor/MANIFEST.json')),
            processor_audit=dict(path=remote_root+'/processor/audit.json',sha256=sha(output/'processor/audit.json')),
            processor_receipts=remote_root+'/processor/inputs.jsonl',cache_root=CACHE,
            conversion_verification=remote_root+'/processor/conversion-verification.json',
            **{key:dict(path=remote_root+'/processor/'+name,sha256=sha(output/'processor'/name))
                for key,name in extras.items() if key!='processor_audit'}),
        pins={**external,**{remote_root+'/'+str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}})
    (output/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    manifest=dict(status='prepared_minds_pair_no_dispatch',image=IMAGE,remote_root=remote_root,
        source_manifest_sha256=spec['source_manifest_sha256'],external_pins=external,
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'PACKAGE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return verify(output,sha(output/'PACKAGE.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('spec','output','remote-root'):p.add_argument('--'+name,required=True)
    a=p.parse_args();result=prepare(read(a.spec),a.output,a.remote_root)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
