"""Re-render actual list context while reusing the verified native acoustic tensors."""
from functools import lru_cache
import torch
from .context_choice import request
from .context_decision_contract import render_prefix_context_request
from .context_reference_dataset import digest
from .span_request_identity import tensor_identity


def identity(x, policy):
    return digest(dict(policy_id=policy,positions=x['positions'][0].tolist(),
                       inputs={k:tensor_identity(v) for k,v in x['inputs'].items()}))


class ChoiceInputs:
    def __init__(self, processor, collate, limit):
        self.processor, self.audio, self.marker, self.limit = (processor, collate['audio_token_id'],
            collate['decision_token_id'], limit)
        self.tokens = lru_cache(maxsize=512)(self._tokens)

    def _tokens(self, questions, requirements, version, n_audio):
        from .context_decision_contract import ContextDecisionRequest
        native=ContextDecisionRequest('unused',questions,requirements,version=version)
        system,history,readouts=render_prefix_context_request(native)
        if history: raise ValueError('single audio only')
        p=self.processor
        messages=[{'role':'system','content':[{'type':'text','text':system}]},
                  {'role':'user','content':[{'type':'audio','audio':''}]},
                  {'role':'assistant','content':readouts}]
        text=p.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        token=p.tokenizer.convert_ids_to_tokens(self.audio)
        if text.count(token)!=1: raise ValueError('one audio placeholder required')
        ids=p.tokenizer(text.replace(token,token*n_audio),return_tensors='pt')['input_ids']
        pos=(ids[0]==self.marker).nonzero().flatten()[None,:]
        aud=(ids[0]==self.audio).nonzero().flatten()
        if (pos.size(1)!=len(questions) or len(aud)!=n_audio or int(aud[-1])>=int(pos[0,0])
                or ids.size(1)>self.limit): raise ValueError('invalid Choice marker or context length')
        return ids,pos

    def apply(self, x, native, row, mode, source, *, indices=None):
        if identity(x,source)!=row['request_key']: raise ValueError('native input differs from Stage1')
        if indices is None:
            indices=row['slate'] if mode=='open' else list(range(len(native.questions)))
        self.render(x,native,mode,indices)
        return indices.index(row['gold']) if row['gold'] in indices else -1

    def render(self,x,native,mode,indices):
        old=x['inputs']
        n=int((old['input_ids']==self.audio).sum())
        def render(req): return self.tokens(req.questions,req.requirements,req.version,n)
        ids,pos=render(native)
        if not torch.equal(ids,old['input_ids']) or not torch.equal(pos,x['positions']) or not old['attention_mask'].eq(1).all():
            raise ValueError('processor text reconstruction differs from authenticated native tensors')
        if mode=='noul': return
        ids,pos=render(request(native,mode,indices))
        new=dict(old,input_ids=ids,attention_mask=torch.ones_like(ids))
        if 'position_ids' in new: new['position_ids']=torch.arange(ids.size(1))[None,:]
        x.update(inputs=new,positions=pos)
