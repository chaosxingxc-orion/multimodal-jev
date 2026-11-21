"""Certify two token templates against independently encoded TRAIN inputs and the native cache."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.encoded_audio_cache import canonical, key_for
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.ordered_prefetch import OrderedPrefetch
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.span_request_identity import tensor_identity
from multimodal_jev.stage1_layout_cache import LayoutTextView, split_text
from multimodal_jev.stage1_training_layout import Stage1TrainingLayout


def checked(path, sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:
        raise ValueError('authenticated input changed: '+str(path))
    return raw


def run(c):
    check_paths({'output':Path(c['output']).parent});torch.set_num_threads(1)
    release=c['clean']; rows,task,_=load(release['root'],release['sha256'],'learning',purpose='optimize')
    d=json.loads(checked(**c['stage1']))
    config=dict(stage='instruction',release_sha256=release['sha256'],stage1_layout=c['layout'],
                instruction_objective=d['stage1']['instruction_objective'])
    layout=Stage1TrainingLayout.open(config,[r['id'] for r in rows],task)
    base=Path(c['native_cache']); m=json.loads(checked(base.parent/'MANIFEST.json',c['native_manifest_sha256']))
    if m['recordings']!=len(rows) or m['release_sha256']!=release['sha256'] or m['schema']!='context-full-learning-cache-v1':
        raise ValueError('native complete learning cache required')
    index={}
    for line in checked(base.parent/'cache.sha256',c['native_index_sha256']).decode().splitlines():
        h,name=line.split();key=name.removeprefix('./').removesuffix('.pt')
        if name!='./'+key+'.pt' or key in index or len(key)!=64:raise ValueError('invalid native cache index')
        index[key]=h
    if len(index)!=len(rows):raise ValueError('native inventory population mismatch')
    panels={};templates={};common=None; seal=None
    for arm,req in layout.requests.items():
        ref=c['examples'][arm];root=Path(ref['root'])
        vm=json.loads(checked(root/'MANIFEST.json',ref['manifest_sha256']))
        if seal is None:seal=vm['seal_sha256']
        if (vm['seal_sha256']!=seal or vm['processor_sha256']!=m['processor_sha256'] or vm['prompt_layout']!=m['prompt_layout']):
            raise ValueError('verified examples must share seal and processor')
        recs=[json.loads(s) for s in checked(root/'inputs.jsonl',vm['inputs_sha256']).splitlines()]
        if len(recs)!=vm['recordings'] or any(r['dataset']!=arm for r in recs):raise ValueError('invalid example roster')
        panels[arm]={r['id']:r for r in recs}
        if not recs or len(panels[arm])!=len(recs):raise ValueError('distinct example IDs required')
        v=torch.load(io.BytesIO(checked(root/'example.pt',recs[0]['cache_sha256'])),map_location='cpu',weights_only=True)
        ident=v['binding'];panel=dict(ident['request']);audio=panel.pop('audio')
        if canonical(req.model_input())!=dict(panel,audio=req.audio) or key_for(ident)!=recs[0]['cache_key']:
            raise ValueError('example not from configured candidate order')
        a,b,_=split_text(v['inputs']['input_ids'][0],vm['encoding']['audio_token_id'])
        templates[arm]=dict(panel=panel,prefix=a,suffix=b)
        static={k:v for k,v in ident.items() if k not in ('request','audio_sha256')}
        if common is None:common=static
        if (common!=static or common['processor_sha256']!=m['processor_sha256'] or
                any(common['context_encoding'][k]!=v for k,v in vm['encoding'].items())):
            raise ValueError('inconsistent example encoding')
    if not panels['native'].keys()==panels['both'].keys()<=set(r['id'] for r in rows):
        raise ValueError('paired examples must be clean TRAIN')
    spec=dict(schema='jev-stage1-layout-text-v1',templates=templates,encoding=common['context_encoding'],
              processor_sha256=m['processor_sha256'],source_identity=release['sha256'],
              layout_plan_sha256=c['layout']['plan']['sha256'],native_manifest_sha256=c['native_manifest_sha256'],
              native_index_sha256=c['native_index_sha256'],examples={a:r['manifest_sha256'] for a,r in c['examples'].items()})
    view=LayoutTextView(spec,layout.requests)
    def one(row):
        def identity(arm):return dict(common,request=dict(templates[arm]['panel'],audio=row['audio']),audio_sha256=row['audio_sha256'])
        native=identity('native');key=key_for(native)
        value=torch.load(io.BytesIO(checked(base/(key+'.pt'),index[key])),weights_only=True,map_location='cpu')
        class Entry:
            def read(self,ident):
                if ident!=value['binding']:raise ValueError('source recording or input binding mismatch')
                return value['inputs'],value['positions'],value['audio_receipt']
        inputs,pos,_=view.read(Entry(),identity('both'))
        if row['id'] in panels['both']:
            for arm,ins,positions in [('native',value['inputs'],value['positions']),('both',inputs,pos)]:
                exp=panels[arm][row['id']]
                if (exp['audio_sha256']!=row['audio_sha256'] or exp['cache_key']!=key_for(identity(arm))
                        or exp['tensors']!={k:tensor_identity(v) for k,v in dict(ins,positions=positions).items()}):
                    raise ValueError('independent processor mismatch: '+row['id']+' '+arm)
        return key,inputs['input_ids'].shape[1]
    seen=set();lo=None;hi=None
    with OrderedPrefetch(one,rows,workers=c['io_workers'],ahead=c['prefetch']) as work:
        for key,n in work:
            seen.add(key);lo=n if lo is None else min(lo,n);hi=n if hi is None else max(hi,n)
    if seen!=index.keys():raise ValueError('full cache key coverage failed')
    spec.update(full_verified=len(seen),panel_verified=len(panels['both']),min_tokens=lo,max_tokens=hi,
                npu_executed=False)
    commit_json(c['output'],spec)
    return {k:spec[k] for k in ('full_verified','panel_verified','min_tokens','max_tokens','npu_executed')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();print(json.dumps(run(load_config(a.config,a.local)[0])))
