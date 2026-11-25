from collections import Counter
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
from multimodal_jev.class_logit_ema import ClassLogitEMA
from multimodal_jev.intent_triplet_sampling import AnchorSampler
from multimodal_jev.intent_triplet_training import OnlineTripletStream, update

CFG = dict(decay=.5, temperature=.2, uniform_mix=.1, min_shared_coordinates=2, max_staleness_updates=1)


def population(labels):
    return {f'{lab}{i}': dict(id=f'{lab}{i}', intent=lab, audio=f'{lab}{i}', audio_sha256='test',
                              official_split='train', role='learning') for lab in labels for i in range(4)}


def test_masked_cosine_full_support_and_real_draws():
    labels = list('abcd')
    ema = ClassLogitEMA(labels, **CFG)
    assert np.allclose(ema.probabilities()[0], [0, 1/3, 1/3, 1/3])
    counts = torch.tensor([[1,2,0,0],[1,1,1,0],[2,1,0,1],[1,0,0,0]])
    means = torch.tensor([[1,-1,0,0],[1,-1,1000,0],[-1,1,0,1000],[1,0,0,0]])
    ema.update(means*counts, counts)
    q = ema.distribution()
    expected = np.exp(np.array([1,-1,0]) / CFG['temperature'])
    assert np.allclose(list(q['a'].values()), .9*expected/expected.sum()+.1/3)
    data = population(labels)
    sampler = AnchorSampler(data,labels,1,42)
    draws = Counter()
    for i in range(1000):
        pos, (neg, (wrong,)) = sampler.draw('a1', 0, i, q)
        assert pos != 'a1' and pos[0] == 'a' and wrong == neg[0] != 'a'
        draws[wrong] += 1
    assert draws['b'] > 850 and draws['c'] and draws['d']
    data['b0']['role'] = 'test'
    with pytest.raises(ValueError, match='TRAIN'):
        AnchorSampler(data,labels,1,42)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.3))
        self.calls = 0

    def forward(self, inputs, positions, **kwargs):
        self.calls += 1
        scores = inputs['x'].float() * self.weight
        if kwargs.get('return_marker'):
            return scores, torch.stack((scores, torch.ones_like(scores)), dim=-1)
        return scores if kwargs.get('return_logits') else scores.sigmoid()

    def clip_grad_norm_(self, value):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), value)


def distributed_steps(rank, path, contrast):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    labels = list('abc')
    data = population(labels)
    config = dict(stage1_runtime={'grad_clip':1.}, mixed={'seed':8}, microbatch=1, triplet_prefetch=dict(workers=1,ahead=1),
                  stop_update=4, class_logit_ema=CFG, instruction_objective=dict(negative_cards_per_anchor=2,
                  temperature=.2, probability_margin=dict(margin=.1, decision_threshold=.5, gap_weight=1., anchor_weight=.25)))
    if contrast:
        config['instruction_objective'] = json.loads(Path(
            'configs/portable/operations/card-stage1-bidirectional-v1.json').read_bytes())['window']['triplet_objective']
    batches = [dict(epoch=i//3, real_queries=2 if i!=1 else 1,
                    slots=[dict(id='a0',weight=1),dict(id='b0',weight=int(i!=1))]) for i in range(4)]

    class Layout:
        assignments = {uid:['r','r'] for uid in data}
        def __call__(self, audio, uid, epoch):
            return audio

    def stream():
        result = OnlineTripletStream(batches,data,lambda item:SimpleNamespace(cache_hit=True,inputs=item[0]),
                    Layout(),dict(labels=labels,catalog=[],arms={'r':labels}),config,rank,2,{})
        result.render = lambda audio,card,uid: dict(x=labels.index(audio[0])+labels.index(card)+1)
        return result

    ema = ClassLogitEMA(labels, **CFG)
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(),lr=.01)
    source = stream()
    try:
        source.begin_epoch(0, ema)
        with patch('multimodal_jev.intent_triplet_training.collate_context_audio',
                   side_effect=lambda rows, **kw: ({'x':torch.tensor([r['x'] for r in rows])},torch.tensor([0]))), \
                patch.object(dist, 'all_reduce', wraps=dist.all_reduce) as reduce:
            for i, version in enumerate([0,0,1]):
                actual, groups, n, used = next(source.queue)
                assert (actual,used) == (i,version)
                update(model,optimizer,groups,n,config,{},'policy','cpu',2,0,ema)
                pending = source.snapshot
                source.publish(ema)
                if i == 1:
                    # Restore at update 2: batch 2 was already prefetched with update 1.
                    replay = stream()
                    replay.begin_epoch(2,ema,pending)
                    try:
                        assert next(replay.queue) == source.queue.pending[0].result()
                    finally:
                        replay.close()
                if i == 0 and not contrast:
                    assert torch.allclose(ema.values[:2],torch.tensor([[.3,.6,.9],[.6,.9,1.2]]))
            assert reduce.call_count == model.calls == 3
        if contrast:
            assert ema.observation_counts.diag().tolist() == [3,2,0]
            assert ema.observation_counts.sum().item() == 5 * 11
        else:
            assert ema.observation_counts.tolist() == [[3,3,3],[2,2,2],[0,0,0]]
        peers = [None,None]
        dist.all_gather_object(peers,ema.receipt(CFG))
        assert peers[0] == peers[1]
        restored = ClassLogitEMA(labels, **CFG)
        restored.load_state_dict(ema.state_dict())
        assert restored.receipt(CFG) == peers[0]
        resumed = stream()
        try:
            source.begin_epoch(3,ema)
            resumed.begin_epoch(3,restored)
            assert next(source.queue) == next(resumed.queue)
        finally:
            resumed.close()
    finally:
        source.close()
        dist.destroy_process_group()


@pytest.mark.parametrize('contrast', [False, True], ids=['triplet', 'card_contrast'])
def test_distributed_prefetch_padding_fused_reduction_and_epoch_restore(tmp_path, contrast):
    torch.multiprocessing.spawn(distributed_steps,(str(tmp_path/'rendezvous'),contrast),nprocs=2)


def test_registered_stage1_commands_use_the_online_contract():
    spec = json.loads(Path('experiments/pipelines/JEV-210.json').read_text())
    train = {s['id']:s for s in spec['stages']['train']}
    first = train['train-stage1-two-epochs']
    assert first['config'] == first['run'][0][1]
    config = json.loads(Path(first['config']).read_text())
    window = config['window']
    assert not {'stop_after_epoch','epoch_mining','bootstrap_input'} & window.keys()
    assert not {'resume-stage1-epoch2','audit-stage1-epoch1'} & train.keys()
    assert window['triplet_prefetch']['ahead'] == window['class_logit_ema']['max_staleness_updates'] == 1
    assert not {'execution_microbatch','effective_anchors_per_rank'} & window.keys()
    assert [b['batch_size_per_rank'] for b in config['token_bucketing']['buckets']] == [32,24,16,8]
    assert config['contrast_samples'] == dict(positive_recordings=1, negative_cards=3, negative_recordings=3)
    assert [s for s in train if s.startswith('score-')] == ['score-stage1-endpoint-train']


def test_epoch2_window_resumes_without_scoring_and_rejects_changed_state(tmp_path):
    from multimodal_jev.baseline import file_hash
    from tools.reports.card_stage1_window import run_resume

    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return file_hash(path)

    entry, work = tmp_path/'entry', tmp_path/'work'
    work.mkdir()
    witness = {'kind':'class-logit-ema-v1', 'updates':3, 'config':CFG}
    ckpt = tmp_path/'checkpoint-0003-committed.json'
    resume = dict(start_update=3, checkpoint_root=str(tmp_path), prepared_root='/prepared',
        checkpoint_sha256=write(ckpt,dict(sampler_state=witness)), sampler_state=witness)
    resume['parent_config_sha256'] = write(entry/'parent-training.json',dict(
        class_logit_ema=CFG, reference_endpoint=6, single_card={}, source_policy='/base',
        pins={}, physical_devices=[0,1], microbatch=1, parity_probability_atol=.01))
    cfg = dict(entry=str(entry),work=str(work),resume=resume,master_addr='localhost',
               master_port=9000,finite_updates=2,parity_passes=4)

    def mock_training_process(argv, check):
        root = Path(argv[-1])
        parent = json.loads((root.parent/'config.json').read_text())
        assert 'resume_confusion' not in parent
        write(root/'result.json',dict(passed=True,start_update=3,stop_update=parent['stop_update'],
                                      engineering_prefix=parent['engineering_prefix']))
        for i in range(2):
            rank = root/f'rank-{i}'
            write(rank/'restored.json',dict(exact_masters=True,exact_adam=True,completed_updates=3))
            write(rank/'sampler-restored.json',witness)
            write(rank/'inference-parity.json',dict(inference_passes=4,compared=4,max_abs=0))
            for s in (4,5):
                write(rank/f'update-{s:04d}.json',{})
    with patch('tools.reports.card_stage1_window.subprocess.run',side_effect=mock_training_process), \
            patch('tools.reports.card_stage1_window.commit_json', side_effect=write), \
            patch('tools.reports.collect_card_first_updates.summarize'):
        assert run_resume(dict(cfg,role='finite'))['stop_update'] == 5
        assert run_resume(dict(cfg,role='formal'))['stop_update'] == 6
    other = tmp_path/'tamper'
    other.mkdir()
    with pytest.raises(ValueError,match='committed online sampler'):
        run_resume(dict(cfg,role='finite',work=str(other),resume=dict(resume,sampler_state={})))
