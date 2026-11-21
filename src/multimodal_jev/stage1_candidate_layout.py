"""Prepare gold-independent Stage1 panel variants; preserve canonical candidate semantics."""
from dataclasses import dataclass, replace
from hashlib import sha256

from .context_decision_contract import render_prefix_context_request


@dataclass(frozen=True)
class CandidateView:
    system: str
    history: str
    readouts: str
    readout_ids: tuple[str, ...]
    canonical_positions: tuple[int, ...]


def build_view(request, *, layout, seed):
    """Return a shared prefix and marker order, plus indices mapping logits back to native order.

    Permutations depend only on candidate IDs and run seed. Caller must gather
    marker logits using canonical_positions before applying the native target.
    """
    layouts={'native', 'definitions', 'readouts', 'both'}
    if layout not in layouts or type(seed) is not int or seed < 0:
        raise ValueError('explicit layout and nonnegative integer seed required')
    canonical=request.questions
    if len(canonical)<2:raise ValueError('at least two candidates required')
    ordered=tuple(sorted(canonical,key=lambda q:(sha256(f'{seed}:{q.id}'.encode()).digest(),q.id)))
    if ordered==canonical:ordered=ordered[1:]+ordered[:1]
    definitions=ordered if layout in ('definitions','both') else canonical
    markers=ordered if layout in ('readouts','both') else canonical
    system,history,_=render_prefix_context_request(replace(request,questions=definitions))
    _,_,readouts=render_prefix_context_request(replace(request,questions=markers))
    ids=tuple(q.id for q in markers)
    mapping={k:i for i,k in enumerate(ids)}
    return CandidateView(system,history,readouts,ids,tuple(mapping[q.id] for q in canonical))


def epoch_layouts(recording_ids, epoch, config):
    """One view per real recording per epoch; rotate assignments without extra exposures."""
    arms=config['layouts']
    if (set(config)!={'layouts','assignment_seed','permutation_seed'} or not isinstance(arms,list)
            or not arms or len(arms)!=len(set(arms)) or 'native' not in arms
            or not set(arms)<={'native','definitions','readouts','both'}
            or any(type(config[k]) is not int or config[k]<0 for k in ('assignment_seed','permutation_seed'))
            or type(epoch) is not int or epoch<0 or not recording_ids
            or len(set(recording_ids))!=len(recording_ids) or any(type(k) is not str or not k for k in recording_ids)):
        raise ValueError('distinct real IDs, nonnegative epoch/seeds and explicit native/variant arms required')
    seed=config['assignment_seed']
    order=sorted(recording_ids,key=lambda k:(sha256(f'{seed}:{k}'.encode()).digest(),k))
    return {k:arms[(i+epoch)%len(arms)] for i,k in enumerate(order)}


def training_request(request, *, layout, seed):
    """Compile a deployable view before target binding or cache lookup (no gold input)."""
    if layout not in ('native','both'):
        raise ValueError('training supports native or synchronized semantic permutation')
    view=build_view(request,layout=layout,seed=seed)
    by_id={q.id:q for q in request.questions}
    compiled=replace(request,questions=tuple(by_id[k] for k in view.readout_ids))
    if render_prefix_context_request(compiled)!=(view.system,view.history,view.readouts):
        raise ValueError('training request does not reproduce the evaluation compiler')
    return compiled
