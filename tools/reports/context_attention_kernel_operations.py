"""Use the shared finite lifecycle for real-input attention kernel diagnosis."""
import argparse
import json
from pathlib import Path
import shutil
from types import FunctionType

import torch

from multimodal_jev.local_checkpoint import commit_json, commit_file
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_acoustic_numeric_operations as shared
from tools.reports.prepare_context_batch_probe_candidate import sha, IMAGE
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.replay_context_attention_capture import normalize_hf
from tools.reports.context_attention_kernel_probe import summarize

FILES = [*shared.FILES, 'tools/reports/context_attention_kernel_probe.py']


def argv(c):
    result = shared.argv(c)
    old = c['remote_root']+'/package/tools/reports/context_acoustic_numeric_probe.py'
    result[result.index(old)] = c['remote_root']+'/package/tools/reports/context_attention_kernel_probe.py'
    return result


def prepare(c, provenance):
    shared.bindings(c)
    source = Path(c['capture_collection']); transport = source.with_suffix('.transport.json')
    if sha(transport) != c['capture_transport_sha256']:
        raise ValueError('capture receipt changed')
    receipt = json.loads(transport.read_text())
    if receipt['status'] != 'transport_complete_independent_audit_required':
        raise ValueError('complete capture transport required')
    def read(name, tensor=False):
        path=source/name
        if sha(path) != receipt['files'][name]: raise ValueError('capture changed: '+name)
        return torch.load(path,map_location='cpu',weights_only=True) if tensor else json.loads(path.read_text())
    original=read('config.json'); rows=[]
    batch=read('evaluation/baseline/batch.pt',True)
    lengths=[r['tensors']['input_ids']['shape'][1] for r in original['rows']]
    for i,n in enumerate(lengths):
        r=normalize_hf(read(f'evaluation/hf/row-{i:02d}.pt',True),n)
        flat=lambda name:r['layer_00.'+name][0].reshape(n,-1,128).contiguous()
        rows.append(dict(q=flat('q_rope'),k=flat('k_rope'),v=flat('v'),expected=flat('kernel_output')))
    if len(rows)!=32 or any(r['q'].shape[1:]!=(16,128) or r['k'].shape[1:]!=(8,128)
            or r['v'].shape!=r['k'].shape for r in rows):raise ValueError('real Qwen3 layer0 layout required')
    for key,name in [('q','q_rope'),('k','k_rope'),('v','v')]:
        captured=torch.cat(batch['layer_00.'+name]).reshape(sum(lengths),-1,128)
        if not torch.equal(torch.cat([r[key] for r in rows]),captured):
            raise ValueError('recorded batch kernel inputs differ')
    payload=dict(rows=rows,recorded_batch=torch.cat(batch['layer_00.kernel_output']).reshape(sum(lengths),16,128))
    dest=Path(c['prepared']);check_paths({'kernel_package':dest});dest.mkdir(exist_ok=False)
    package=dest/'package';package.mkdir()
    for name in FILES:
        target=package/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(c['source'])/name,target)
    commit_file(package/'inputs.pt',lambda f:torch.save(payload,f))
    config=dict(rows=[dict(id=r['id']) for r in original['rows']],lengths=lengths,
        capture_transport_sha256=c['capture_transport_sha256'],model=c['model'],cache_root=c['cache_root'])
    commit_json(package/'config.json',config)
    commit_json(package/'FILES.json',{str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()})
    result=dict(status='sealed_attention_kernel_probe',image=IMAGE,argv=argv(c),files_sha256=verify(package),
        provenance=provenance,training_dispatch_allowed=False)
    commit_json(dest/'PREPARED.json',result);return result


def validate(c):
    shared.bindings(c);root=Path(c['prepared']);r=json.loads((root/'PREPARED.json').read_text())
    if (r['status']!='sealed_attention_kernel_probe' or r['files_sha256']!=verify(root/'package')
            or r['image']!=IMAGE or r['argv']!=argv(c)):
        raise ValueError('sealed kernel-only probe required')
    for name in FILES:
        if sha(root/'package'/name)!=sha(Path(c['source'])/name):raise ValueError('probe source changed')
    return r,json.loads((root/'package/config.json').read_text())


def audit(c,collection,handle):
    fn=FunctionType(shared.audit.__code__,dict(shared.audit.__globals__,validate=validate),'audit')
    result=fn(c,collection,handle);root=Path(collection)/'evaluation'
    payload=torch.load(Path(c['prepared'])/'package/inputs.pt',map_location='cpu',weights_only=True)
    outputs=torch.load(root/'outputs.pt',map_location='cpu',weights_only=True)
    if summarize(payload,outputs)!=json.loads((root/'report.json').read_text())['comparisons']:
        raise ValueError('independent kernel replay comparison differs')
    return dict(result,status='attention_kernel_independently_compared_not_admission')


def launch(c,provenance):
    ns=shared.namespace();ns.update(validate=validate,audit=audit)
    return ns['launch'](c,provenance)


if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    r=prepare(c,provenance) if a.action=='prepare' else validate(c)[0] if a.action=='validate' else launch(c,provenance)
    print(json.dumps(r,indent=2))
