"""One independently attended sequence per (audio, card); combine logits only for the loss."""
import json
from pathlib import Path

import torch

from .baseline import file_hash
from .context_decision_contract import ContextDecisionRequest, ApplicabilityQuestion
from .context_audio_collate import forward_context_audio_batch
from .stage1_layout_cache import split_text


def request(card, rules, cfg):
    return ContextDecisionRequest('audio',(ApplicabilityQuestion(card['id'],card['definition'],cfg['question']),),
        tuple([cfg['instruction'],*rules]),version=cfg['version'])


def prepare(entry, output, policy, sample, cfg):
    from transformers import Qwen3ASRProcessor
    from .audio_policy import load_audio
    from .context_decision_contract import encode_context_request
    from .local_checkpoint import commit_json
    if file_hash(sample['path'])!=sample['sha256']:
        raise ValueError('single-card encoding witness changed')
    plan=json.loads((entry/'card-plan.json').read_bytes())
    bank=json.loads((entry/'sampled-bank.json').read_bytes())
    native=json.loads((entry/'native-cache.json').read_bytes())
    processor=Qwen3ASRProcessor.from_pretrained(str(policy/'processor'),local_files_only=True)
    audio=load_audio(sample['path'],None)
    enc={k:native['encoding'][k] for k in ('decision_token_id','audio_token_id','max_tokens','prompt_layout')}
    ids=[card['id'] for card in bank['cards']]
    rules={i:[r['rule'] for r in bank['semantic_rules'] if i in r['labels']] for i in ids}
    catalog,indices,arms=[],{},{}
    for arm,p in plan['panels'].items():
        if arm=='native':continue
        if [q['id'] for q in p['questions']]!=ids:raise ValueError('changed loss candidate order')
        arms[arm]=[]
        for q in p['questions']:
            key=q['id'],q['definition']
            if key not in indices:
                r=request(q,rules[q['id']],cfg)
                batch,pos=encode_context_request(processor,r,audio,**enc)
                before,after,_=split_text(batch['input_ids'][0],enc['audio_token_id'])
                if pos.numel()!=1 or after.count(enc['decision_token_id'])!=1:
                    raise ValueError('exactly one marker after the audio required')
                indices[key]=len(catalog)
                catalog.append(dict(id=q['id'],prefix=before,suffix=after))
            arms[arm].append(indices[key])
    commit_json(output,dict(arms=arms,catalog=catalog,labels=ids,config=cfg,plan=file_hash(entry/'card-plan.json')))
    return dict(classes=len(ids),unique_prompts=len(catalog),questions_per_sequence=1)


class IndependentForward:
    def __init__(self,spec,assignments):
        labels=spec['labels']
        self.arms=spec['arms'];self.assignments=assignments
        self.tokens=[(torch.tensor(c['prefix']),torch.tensor(c['suffix'])) for c in spec['catalog']]
        if not all([spec['catalog'][i]['id'] for i in arm]==labels for arm in self.arms.values()):
            raise ValueError('loss columns must follow the frozen label order')
        self.width=len(labels)

    def __call__(self,model,items,**kwargs):
        rows=[]
        for item in items:
            # The full cached view is an authenticated transport of acoustic features/length.
            # Its instruction tokens are never submitted to the model.
            uid,exposure=item['request_id'].rsplit('::epoch',1)
            epoch=int(exposure.split('-',1)[0])
            arm=self.assignments[uid][epoch]
            acoustic=item['inputs']
            tokens=acoustic['input_ids'][0]
            sound=tokens[tokens==kwargs['audio_token_id']]
            if not len(sound):raise ValueError('missing acoustic sequence')
            for idx in self.arms[arm]:
                prefix,suffix=self.tokens[idx]
                ids=torch.cat((prefix,sound,suffix))[None,:]
                positions=(ids[0]==kwargs['decision_token_id']).nonzero().T.contiguous()
                if positions.shape!=(1,1) or int(positions[0,0])<len(prefix)+len(sound):
                    raise ValueError('each model row needs one post-audio marker')
                inp=dict(input_ids=ids,attention_mask=torch.ones_like(ids),
                         input_features=acoustic['input_features'],input_features_mask=acoustic['input_features_mask'])
                rows.append(dict(request_id=item['request_id']+':card'+str(idx),policy_id=item['policy_id'],
                                 inputs=inp,positions=positions))
        scores=forward_context_audio_batch(model,rows,**kwargs)
        if scores.shape!=(len(rows),1):raise ValueError('one logit per independent sequence required')
        return scores.reshape(len(items),self.width)
