"""Read-only full selection cache verification and concrete marker receipts.

No audio decode, processor, network, weights, labels or NPU. Uses torch CPU to
recompute tensor identities against the previously authenticated full evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.span_request_identity import tensor_identity


def verify(binding_root,cache_root,output,*,binding_sha256,decision_token_id=151706,audio_token_id=151676):
    binding_root=Path(binding_root);cache_root=Path(cache_root);output=Path(output)
    if file_hash(binding_root/'binding.json')!=binding_sha256:raise ValueError('input binding changed')
    binding=json.loads((binding_root/'binding.json').read_text())
    expected_names={'population.json','candidate-ids.json','cache-receipts.json'}
    if set(binding['artifacts'])!=expected_names or binding['recordings']!=9226:
        raise ValueError('complete selection evidence package required')
    for name,h in binding['artifacts'].items():
        if file_hash(binding_root/name)!=h:raise ValueError('input evidence changed')
    def read(name):return json.loads((binding_root/name).read_text())
    population=read('population.json');candidates=read('candidate-ids.json');prior=read('cache-receipts.json')
    if (len(population)!=9226 or len(prior)!=9226 or len({r['id'] for r in population})!=9226
            or len(candidates)!=60 or len(set(candidates))!=60 or not cache_root.is_dir()):
        raise ValueError('full unique selection/candidates and existing cache required')
    check_paths({'output':output});output.mkdir(parents=True,exist_ok=False)
    cache=EncodedAudioCache(cache_root);receipts=[];started=time.monotonic()
    try:
        for index,(asset,evidence) in enumerate(zip(population,prior,strict=True)):
            if any(evidence[k]!=v for k,v in asset.items()):raise ValueError('cache/population order differs')
            key=evidence['cache_key']
            if len(key)!=64 or any(c not in '0123456789abcdef' for c in key):raise ValueError('safe content-addressed cache key required')
            path=cache_root/(key+'.pt')
            if path.is_symlink() or file_hash(path)!=evidence['cache_sha256']:raise ValueError('actual cache bytes differ')
            value=torch.load(path,map_location='cpu',weights_only=True)
            identity=value['binding']
            request_hash=hashlib.sha256((json.dumps(identity['request'],sort_keys=True,ensure_ascii=False)+'\n').encode()).hexdigest()
            if (key_for(identity)!=key or identity['schema']!='encoded-audio-cache-v1'
                    or identity['audio_sha256']!=asset['audio_sha256'] or request_hash!=asset['request_sha256']
                    or identity['processor_sha256']!=binding['processor_sha256'] or identity['preprocessing'] is not None
                    or identity['context_encoding']!=dict(decision_token_id=decision_token_id,audio_token_id=audio_token_id,
                        max_tokens=4096,features_dtype='bfloat16',version='context-cache-v1')):
                raise ValueError('cache request/processor/encoding binding differs')
            # Read through the established immutable-cache validator as well.
            inputs,positions,audio=cache.read(identity)
            actual={k:tensor_identity(v) for k,v in dict(inputs,positions=positions).items()}
            if actual!=evidence['tensors']:raise ValueError('actual tensor differs from independent evidence')
            if (positions.shape!=(1,60) or inputs['input_ids'].shape[1]>4096
                    or inputs['input_features'].dtype!=torch.bfloat16
                    or not torch.equal((inputs['input_ids'][0]==decision_token_id).nonzero().flatten(),positions[0])):
                raise ValueError('complete marker positions and BF16 features required')
            receipts.append(dict(asset,candidate_ids=candidates,positions=positions[0].tolist(),
                inputs={k:v for k,v in actual.items() if k!='positions'},audio_receipt=audio))
            if (index+1)%256==0:atomic_json(output/'progress.json',dict(completed=index+1,total=9226,seconds=time.monotonic()-started))
        commit_json(output/'input-receipts.json',receipts)
        report=dict(recordings=9226,decisions=60,binding_sha256=binding_sha256,
            input_receipts_sha256=file_hash(output/'input-receipts.json'),seconds=time.monotonic()-started,
            scope='All actual cache bytes/tensors and marker positions verified; no waveform recomputation, NPU inference or task score')
        commit_json(output/'result.json',report);commit_json(output/'terminal.json',dict(exit_code=0))
        return report
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only verification required')
    torch.set_num_threads(1)
    print(json.dumps(verify(**config),indent=2))
