"""Stage1: reuse authenticated acoustic features across two fixed candidate orders."""
from pathlib import Path
import hashlib
import json

import torch

from .encoded_audio_cache import canonical
from .context_decision_contract import PREFIX_LAYOUT


def split_text(ids, audio_id):
    loc = (ids == audio_id).nonzero().flatten()
    if not len(loc) or len(loc) != int(loc[-1] - loc[0]) + 1:
        raise ValueError('one contiguous audio expansion required')
    return ids[:loc[0]].tolist(), ids[loc[-1]+1:].tolist(), len(loc)


class LayoutTextView:
    def __init__(self, spec, requests):
        card_views = spec['schema'] == 'jev-card-text-cache-v1'
        if not (card_views or (spec['schema'] == 'jev-stage1-layout-text-v1' and set(spec['templates']) == {'native','both'})) or set(requests) != set(spec['templates']):
            raise ValueError('two verified text templates required')
        self.spec = spec
        self.panels = {}
        self.templates = {}
        for arm, request in requests.items():
            panel = canonical(request.model_input())
            del panel['audio']
            if panel != spec['templates'][arm]['panel']:
                raise ValueError('template and sealed task disagree')
            self.panels[arm] = panel
            t = spec['templates'][arm]
            self.templates[arm] = tuple(torch.tensor(t[k], dtype=torch.long) for k in ('prefix','suffix'))
        if 'native' not in self.panels or (not card_views and self.panels['native']==self.panels['both']):
            raise ValueError('distinct complete task views required')

    def read(self, base, identity):
        encoding = identity['context_encoding']
        if (encoding != self.spec['encoding'] or encoding['prompt_layout'] != PREFIX_LAYOUT
                or self.spec['processor_sha256'] != identity['processor_sha256']):
            raise ValueError('layout cache processor or encoding mismatch')
        request = dict(identity['request'])
        audio = request.pop('audio')
        arm = next((k for k,v in self.panels.items() if v == request), None)
        if arm is None:
            raise ValueError('unknown candidate layout in cached request')
        native = dict(identity, request=dict(self.panels['native'],audio=audio))
        inputs, pos, receipt = base.read(native)
        if (set(inputs) != {'input_ids','attention_mask','input_features','input_features_mask'}
                or not torch.all(inputs['attention_mask']==1)):
            raise ValueError('one unpadded complete request required')
        prefix,suffix = self.templates['native']
        ids = inputs['input_ids'][0]
        n = len(ids)-len(prefix)-len(suffix)
        if (n<=0 or not len(suffix) or not torch.equal(ids[:len(prefix)],prefix)
                or not torch.equal(ids[-len(suffix):],suffix)
                or not torch.all(ids[len(prefix):len(prefix)+n]==encoding['audio_token_id'])
                or not torch.equal(pos,(ids==encoding['decision_token_id']).nonzero().T)):
            raise ValueError('native cache does not reproduce authenticated processor template')
        if arm=='native':
            return inputs,pos,receipt
        before,after=self.templates[arm]
        ids=torch.cat((before,ids[len(prefix):len(prefix)+n],after))[None,:]
        positions=(ids[0]==encoding['decision_token_id']).nonzero().T.contiguous()
        if (ids.shape[1]>encoding['max_tokens'] or positions.shape!=pos.shape
                or int(positions.min())<len(before)+n):
            raise ValueError('permuted marker count, order or token budget differs')
        return dict(inputs,input_ids=ids,attention_mask=torch.ones_like(ids)),positions,receipt


def read_view(pin, layout, config):
    p=Path(pin['path']);raw=p.read_bytes()
    if p.is_symlink() or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
        raise ValueError('layout cache proof changed')
    d=json.loads(raw)
    if (d['layout_plan_sha256']!=config['stage1_layout']['plan']['sha256']
            or d['source_identity']!=config['release_sha256'] or d['panel_verified']<=0
            or d['full_verified']!=config['batch_config']['source_queries']
            or d['native_manifest_sha256']!=config['pins'].get(str(Path(config['encoded_audio_cache']).parent/'MANIFEST.json'))):
        raise ValueError('matching full-population layout cache proof required')
    return LayoutTextView(d,layout.requests)


class ViewCache:
    def __init__(self,base,view):
        self.base=base
        self.view=view

    def read(self,identity):
        return self.view.read(self.base,identity)
