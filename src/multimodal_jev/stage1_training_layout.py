"""Bind the frozen layout plan to the full learning population before prefetch."""
from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json

from .context_candidate_task import compile_candidates
from .slurp_training import content_hash
from .stage1_candidate_layout import build_view, epoch_layouts, training_request


class Stage1TrainingLayout:
    def __init__(self, ids, task, *, design, plan, release_sha256, objective):
        self.assignments=[]
        if (design['schema']!='jev-next-simple-two-stage-v1' or design['training']['layouts']!=['native','both']
                or plan['source_identity']!=release_sha256 or plan['stage1_objective']!=objective
                or set(plan['arms'])!={'training'} or set(plan['panels'])!={'native','both'}):
            raise ValueError('single complete native/permuted Stage1 plan required')
        cfg=design['training']; witnesses=plan['arms']['training']
        if len(witnesses['epochs'])!=2 or witnesses['real_exposures']!=2*len(ids):
            raise ValueError('full matched two-epoch population required')
        for epoch,witness in enumerate(witnesses['epochs']):
            assignment=epoch_layouts(ids,epoch,cfg)
            if content_hash(assignment)!=witness['assignment_sha256']:
                raise ValueError('sealed per-recording epoch assignment differs')
            self.assignments.append(assignment)
        native=compile_candidates('audio',[dict(id=q['id'],definition=q['description'])
            for q in task['schema']['candidates']],task['thinking']['requirements'],semantics='exclusive_event')
        self.requests={}
        for arm,expected in plan['panels'].items():
            view=build_view(native,layout=arm,seed=cfg['permutation_seed'])
            if content_hash(asdict(view))!=content_hash(expected):
                raise ValueError('current candidate semantics differ from the sealed panel')
            self.requests[arm]=training_request(native,layout=arm,seed=cfg['permutation_seed'])

    def __call__(self,audio,uid,epoch):
        if type(epoch) is not int or not 0<=epoch<len(self.assignments):
            raise ValueError('invalid Stage1 epoch')
        return replace(self.requests[self.assignments[epoch][uid]],audio=audio)

    @classmethod
    def open(cls, config, ids, task):
        spec=config['stage1_layout']
        if set(spec)!={'design','plan'} or config['stage']!='instruction':
            raise ValueError('explicit layout design and plan required for Stage1')
        def read(pin):
            p=Path(source_pin(config,pin)['path']);raw=p.read_bytes()
            if p.is_symlink() or hashlib.sha256(raw).hexdigest()!=pin['sha256']:
                raise ValueError('pinned Stage1 layout changed')
            return json.loads(raw)
        design,plan=(read(spec[k]) for k in ('design','plan'))
        if plan['design_sha256']!=spec['design']['sha256'] or plan['stage1_design_sha256']!=design['stage1']['sha256']:
            raise ValueError('layout plan belongs to a different design')
        return cls(ids,task,design=design,plan=plan,release_sha256=config['release_sha256'],
                   objective=config['instruction_objective'])


def source_pin(config, pin):
    """Internal pins move with the source tree at finite, replay, tail and formal stages."""
    if Path(pin['path']).is_absolute():
        return pin
    from .source_snapshot import regular
    return dict(pin,path=str(regular(Path(config['source_root']),pin['path'])))


def checkpoint_layout(config):
    """Paths may relocate; the complete layout protocol may not change on resume."""
    spec=config['stage1_layout']
    return dict(design=spec['design']['sha256'],plan=spec['plan']['sha256'],cache=config['layout_cache']['sha256'])
