"""Bounded prefix hidden-state companion to a paged attention KV cache.

KV states alone cannot restore a history-attending Jev head. Store final
backbone states per prefix block, bound to policy/runtime/token/audio identity.
A missing sidecar block fails closed; never substitute zeros or output cache.
"""
from collections import OrderedDict
import hashlib,json
import torch


def block_keys(tokens,*,policy_id,runtime_id,audio_id,audio_start,block_size):
    if (not tokens or any(type(t)is not int or t<0 for t in tokens) or
        not all(isinstance(v,str) and v for v in (policy_id,runtime_id,audio_id)) or
        type(audio_start)is not int or not 0<=audio_start<len(tokens) or
        type(block_size)is not int or block_size<=0):raise ValueError('complete prefix identity required')
    seed=hashlib.sha256(json.dumps([policy_id,runtime_id,block_size]).encode()).hexdigest();keys=[]
    for begin in range(0,len(tokens)-block_size+1,block_size):
        end=begin+block_size
        seed=hashlib.sha256(json.dumps([seed,tokens[begin:end],audio_id if end>audio_start else None],separators=(',',':')).encode()).hexdigest()
        keys.append(seed)
    return keys


class PrefixHistory:
    def __init__(self,*,block_size,max_bytes):
        if type(block_size)is not int or block_size<1 or type(max_bytes)is not int or max_bytes<1:raise ValueError('positive cache budgets required')
        self.block_size=block_size;self.max_bytes=max_bytes;self.blocks=OrderedDict();self.bytes=0;self.hit_tokens=0;self.miss_blocks=0

    def keys(self,tokens,binding):return block_keys(tokens,block_size=self.block_size,**binding)

    def restore(self,suffix,tokens,computed,binding):
        if suffix.ndim!=2 or type(computed)is not int or computed<0 or computed%self.block_size or computed+len(suffix)!=len(tokens):
            raise ValueError('complete prefill suffix and block-aligned cache hit required')
        keys=self.keys(tokens,binding)[:computed//self.block_size];parts=[]
        for key in keys:
            if key not in self.blocks:
                self.miss_blocks+=1;raise ValueError('KV hit has no matching head-history block')
            part=self.blocks[key]
            if part.dtype!=suffix.dtype or part.device!=suffix.device or part.shape!=(self.block_size,suffix.shape[1]):raise ValueError('history shape/runtime changed')
            self.blocks.move_to_end(key);parts.append(part)
        self.hit_tokens+=computed
        full=torch.cat([*parts,suffix],0) if parts else suffix
        for index,key in enumerate(self.keys(tokens,binding)):
            block=full[index*self.block_size:(index+1)*self.block_size]
            if key in self.blocks:
                # Same identity must never produce a different backbone state.
                if not torch.equal(self.blocks[key],block):raise ValueError('prefix hidden states changed for identical cache identity')
                self.blocks.move_to_end(key);continue
            stored=block.detach().clone();size=stored.numel()*stored.element_size()
            if size>self.max_bytes:raise ValueError('history budget smaller than one block')
            while self.bytes+size>self.max_bytes:
                _,old=self.blocks.popitem(last=False);self.bytes-=old.numel()*old.element_size()
            self.blocks[key]=stored;self.bytes+=size
        return full

    def clear(self):self.blocks.clear();self.bytes=0
