import torch
from torch import nn
from types import SimpleNamespace
from multimodal_jev.card_training_acoustics import collate,hidden
from multimodal_jev.context_audio_collate import collate_context_audio

def test_card_training_dedup_retains_order_padding_and_gradients():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower=nn.Identity()
            self.multi_modal_projector=nn.Identity()
            self.embed=nn.Embedding(16,1)
            self.scale=nn.Parameter(torch.tensor(0.2))
        def get_input_embeddings(self):return self.embed
        def get_audio_features(self,x,m,return_dict):
            return SimpleNamespace(pooler_output=torch.cat([r[0,:int(w.sum()):8] for r,w in zip(x,m)])[:,None])
        def get_placeholder_mask(self,ids,emb,sound):return (ids==8).unsqueeze(-1)
        def forward(self,inputs_embeds,**kw):return SimpleNamespace(last_hidden_state=self.scale*inputs_embeds.cumsum(1))
    model=Model().eval()
    owners=[(torch.arange(100).float()[None,None,:]+i,(torch.arange(100)<n)[None].long()) for i,n in [(1,8),(2,16)]]
    rows=[]
    for i,j in enumerate((1,0,1,0)):
        f,m=owners[j];ids=torch.tensor([[2+i,*([8]*(j+1)),9]])
        rows.append(dict(request_id=str(i),policy_id='p',positions=torch.tensor([[ids.numel()-1]]),inputs=dict(
            input_ids=ids,attention_mask=torch.ones_like(ids),input_features=f,input_features_mask=m)))
    opts=dict(audio_token_id=8,decision_token_id=9,pad_token_id=0,audio_chunk_size=100,audio_infer_window=100)
    base,pos=collate_context_audio(rows,policy_id='p',**opts);new,pos2=collate(rows,policy_id='p',**opts)
    assert new['input_features'].shape[0]==2 and torch.equal(pos,pos2)
    x=model.embed(base['input_ids']);a=model.get_audio_features(base['input_features'],base['input_features_mask'],True).pooler_output
    ref=model(x.masked_scatter((base['input_ids']==8).unsqueeze(-1),a)).last_hidden_state
    out=hidden(model,new)
    assert torch.equal(ref,out)
    for a,b in zip(torch.autograd.grad(ref.sum(),model.parameters()),torch.autograd.grad(out.sum(),model.parameters())):
        assert torch.equal(a,b)
