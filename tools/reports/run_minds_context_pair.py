"""Label-free MINDS native-descriptor pair; both parity panels precede either full inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.jev.prepare_fsc_context_source import verify as verify_source
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.infer_fsc_context_transfer import validate_source
from tools.reports.minds_context_runtime import load_inputs, require_parity, checked_prediction


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def configs(spec):
    source=Path(spec['source_package'])
    manifest=verify_source(source,spec['source_manifest_sha256'])
    if spec['physical_devices']!=[7]:raise ValueError('isolated device7 required')
    expected={'inputs','input_manifest_sha256','conversion_verification','protocol',
              'processor_manifest','processor_audit','processor_receipts','cache_root',
              'preparation_transport','verification_transport','actual_verification'}
    if set(spec['inputs'])!=expected:raise ValueError('only label-free MINDS artifacts allowed')
    for p,h in spec['pins'].items():
        if sha(p)!=h:raise ValueError('runtime source changed: '+p)
    pair={}
    for role in ('initial','selected'):
        entry=manifest['policies'][role]
        c=dict(spec['inputs'],pins=spec['pins'],model=spec['model'],base_id=entry['contract']['base_id'],
            snapshot=str(source/entry['policy']),adapter=str(source/entry['adapter']),
            policy_id=entry['policy_id'],contract=entry['contract'],source_role=role,
            source_selection=dict(path=str(source/'source-selection.json'),sha256=sha(source/'source-selection.json')),
            physical_devices=[7])
        validate_source(c)
        pair[role]=c
    rows,receipts,_,processor=load_inputs(pair['initial'])
    if processor['processor_sha256']!=manifest['processor_sha256']:
        raise ValueError('frozen source and MINDS processor differ')
    return pair,rows,receipts


def check_full(folder,c,rows,receipts):
    result=read(folder/'result.json');predictions=read(folder/'predictions.json')['records']
    if (len(predictions)!=len(rows) or result['recordings']!=8168 or result['decisions']!=14
            or result['policy_id']!=c['policy_id'] or result['source_role']!=c['source_role']
            or result['input_manifest_sha256']!=c['input_manifest_sha256']
            or result['predictions_sha256']!=sha(folder/'predictions.json')):
        raise ValueError('complete bound MINDS inference required')
    for row,receipt,prediction in zip(rows,receipts,predictions,strict=True):
        if prediction['id']!=row['id']:raise ValueError('MINDS population order differs')
        checked_prediction(row,receipt,prediction,c['policy_id'])
    return result


def invoke(cpath,folder,phase):
    with (folder/(phase+'.log')).open('x') as log:
        subprocess.run([sys.executable,'-m','tools.reports.minds_context_runtime',
            '--config',str(cpath),'--output',str(folder),'--phase',phase],
            stdout=log,stderr=subprocess.STDOUT,check=True)


def run(spec,output):
    pair,rows,receipts=configs(spec)
    output=Path(output).resolve()
    for path in (spec['source_package'],spec['model'],spec['inputs']['inputs'],spec['inputs']['cache_root']):
        root=Path(path).resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):raise ValueError('independent output required')
    from multimodal_jev.local_runtime_storage import require_training_storage
    require_training_storage(output);output.mkdir(exist_ok=False)
    write(output/'input.json',spec)
    phase='prepared'
    try:
        for role,c in pair.items():
            folder=output/role;folder.mkdir();write(folder/'config.json',c)
        panels={}
        for role,c in pair.items():
            folder=output/role
            for backend in ('hf','vllm'):
                phase=role+'-'+backend;invoke(folder/'config.json',folder,backend)
            panels[role]=require_parity(c,folder,rows,receipts)
            if panels[role]!=read(folder/'parity.json'):raise ValueError('parity serialization differs')
        results={}
        for role,c in pair.items():
            phase=role+'-full-inference';folder=output/role
            invoke(folder/'config.json',folder,'inference')
            results[role]=check_full(folder,c,rows,receipts)
        report=dict(status='minds_pair_inference_complete_independent_collection_pending',
            source_manifest_sha256=spec['source_manifest_sha256'],panels=panels,policies=results,
            target_tuning=False,scope='Label-free inference; independent collection and scoring remain required')
        write(output/'result.json',report)
        write(output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
        return report
    except BaseException as exc:
        write(output/'terminal.json',dict(exit_code=1,phase=phase,error=repr(exc),finished_unix=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read(a.spec),a.output)
