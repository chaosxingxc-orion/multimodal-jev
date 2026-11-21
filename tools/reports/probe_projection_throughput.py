"""Finite NPU projection study: identical BF16 values and bounded memory.

No trained weights, task labels or official evaluation. Four passes per shape,
including warmup. This is not full-model speed or numerical admission.
"""
import argparse,json,time
from pathlib import Path
import torch
import torch_npu
from multimodal_jev.span_fixed_shape_linear import fixed_shape_linear
from multimodal_jev.local_checkpoint import commit_json


def vectorized(x,w,*,tile_rows=128,tile_batch=8):
    b,n,k=x.shape;padded=((n+tile_rows-1)//tile_rows)*tile_rows
    tiles=torch.nn.functional.pad(x,(0,0,0,padded-n)).reshape(-1,tile_rows,k)
    count=len(tiles);tiles=torch.nn.functional.pad(tiles,(0,0,0,0,0,(-count)%tile_batch))
    y=torch.bmm(tiles,w.t()[None].expand(len(tiles),-1,-1))
    return y[:count].reshape(b,padded,w.shape[0])[:,:n]


def main(c):
    torch.set_num_threads(2);torch.npu.set_device(0);torch.manual_seed(c['seed']);torch.npu.manual_seed_all(c['seed'])
    out=Path(c['output']);out.mkdir(exist_ok=False);commit_json(out/'config.json',c);results=[]
    for n,k,d in c['shapes']:
        x=torch.randn((1,n,k),device='npu',dtype=torch.bfloat16);w=torch.randn((d,k),device='npu',dtype=torch.bfloat16)
        row={'shape':[n,k,d],'modes':{}};reference=None
        for name,fn in [('fixed8_loop',fixed_shape_linear),('all_tiles_bmm',vectorized)]:
            times=[];exact=True;maximum=0.
            for i in range(4):
                torch.npu.synchronize();start=time.perf_counter();y=fn(x,w);torch.npu.synchronize();times.append(time.perf_counter()-start)
                if name=='fixed8_loop':reference=y.clone()
                else:
                    exact=exact and torch.equal(y,reference);maximum=max(maximum,(y-reference).abs().max().item())
            row['modes'][name]={'seconds':times,'exact':exact,'max_abs':maximum,'peak_allocated':torch.npu.max_memory_allocated()}
        a=row['modes'];row['warm_speedup']=sum(a['fixed8_loop']['seconds'][1:])/sum(a['all_tiles_bmm']['seconds'][1:]);results.append(row)
        commit_json(out/f'shape-{len(results):02d}.json',row)
        del x,w,y,reference
    commit_json(out/'result.json',{'results':results,'full_model_admitted':False})
    commit_json(out/'terminal.json',{'exit_code':0,'full_model_admitted':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(json.loads(Path(a.config).read_text()))
