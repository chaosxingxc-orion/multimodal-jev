"""Reuse fixed ontology structure while retaining the original input checker."""
import json
from types import FunctionType,SimpleNamespace
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import DECISION_TOKEN,RESERVED_TIMESTAMP_TOKEN
from tools.reports import context_reference_input_evidence as original


def compiler():
    template=None;binding=None
    def compile(audio,candidates,requirements,*,semantics,history=()):
        nonlocal template,binding
        identity=(candidates,requirements,semantics,history)
        if template is None or identity!=binding:
            request=compile_candidates(audio,candidates,requirements,semantics=semantics,history=history)
            template=request.model_input()
            import copy
            binding=copy.deepcopy(identity)
        else:
            if type(audio)is not str or not audio.strip():raise ValueError('audio required')
            encoded=json.dumps(audio,ensure_ascii=False)
            if any(t in encoded for t in (DECISION_TOKEN,RESERVED_TIMESTAMP_TOKEN)):
                raise ValueError('reserved decision marker in request')
        # Fresh outer map; nested ontology is private and consumed read-only by
        # the original binder's content_hash. No object escapes this scope.
        return SimpleNamespace(model_input=lambda:dict(template,audio=audio))
    return compile


def bind(proof,stream,**arguments):
    fn=FunctionType(original.bind.__code__,dict(original.bind.__globals__,compile_candidates=compiler()))
    return fn(proof,stream,**arguments)
