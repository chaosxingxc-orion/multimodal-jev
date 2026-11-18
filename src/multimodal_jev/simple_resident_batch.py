"""Collect scheduled RL batches with one resident engine and one audio compile.

The owner authenticates the runtime, learning release and schedule before use.
This is behavior transport, not optimizer execution or training admission.
Only label-free renderer rows enter the engine; targets are joined afterwards.
"""
import copy
import time

from .simple_audio_batch import compile_batch, renderer_for_policy
from .simple_sampled_rollout import actor_batch


class ResidentSimpleBatch:
    def __init__(self, engine, population, task, processor, reference_config, *, seed):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('explicit sampling seed required')
        self.engine = engine
        self.population = population
        self.task = task
        self.processor = processor
        self.reference = copy.deepcopy(reference_config)
        self.reference_id = self.reference['base_id']+':'+self.reference['adapter_id']
        self.seed = seed
        self.phase = 'ready'
        self.last_update = None
        self.last_renderer = None
        self.last_behavior = None
        self.last_reference = None

    def collect(self, slots, *, policy_config, completed_updates):
        """Return CPU actor items for the next optimizer update, without HF.

        Repeated/out-of-order cursors are rejected; a failed call poisons the
        collector. Resume creates a fresh owner using an authenticated cursor.
        The owner must publish/authenticate the new actor adapter before each
        call. Equal BF16 adapter bytes on adjacent updates are allowed.
        """
        if self.phase != 'ready':
            raise ValueError('ready collector required; uncertain work cannot be retried')
        self.phase = 'collecting'
        try:
            result = self._collect(slots, policy_config, completed_updates)
        except BaseException:
            self.phase = 'failed'
            raise
        self.last_update = completed_updates
        self.phase = 'ready'
        return result

    def _collect(self, slots, config, update):
        if (type(update) is not int or update < 0
                or (self.last_update is not None and update != self.last_update+1)):
            raise ValueError('exact consecutive optimizer cursor required')
        fields = ('model', 'base_id', 'runtime_profile', 'max_model_len', 'physical_devices')
        if any(config[k] != self.reference[k] for k in fields):
            raise ValueError('behavior and fixed reference runtime differ')
        policy = config['base_id']+':'+config['adapter_id']
        start = time.monotonic()
        entries, renderer, compiled = compile_batch(self.population, self.task, slots,
            self.processor, policy_id=policy, global_batch=128)
        compile_seconds = time.monotonic()-start
        # Per-batch expected count includes unique padding inputs when present.
        current = dict(config, population={'requests': len(renderer)})
        fixed = dict(self.reference, population={'requests': len(renderer)})
        start = time.monotonic()
        if policy == self.reference_id:
            # Stage boundary (or unchanged BF16 export): same exact immutable
            # model/support, so this distribution serves both roles.
            behavior = self.engine.rollout(current, renderer)
            reference = behavior
            passes = 1
        else:
            reference = self.engine.rollout(fixed,
                renderer_for_policy(renderer, self.reference_id))
            behavior = self.engine.rollout(current, renderer)
            passes = 2
        rollout_seconds = time.monotonic()-start
        start = time.monotonic()
        items, sampling = actor_batch(entries, behavior, reference, policy_id=policy,
            reference_policy_id=self.reference_id, seed=self.seed, update=update)
        # Bounded one-batch tensors for post-update export/actor parity. Reuse
        # the very same processed audio, then replace on the next collect.
        self.last_renderer = renderer
        self.last_behavior = behavior
        self.last_reference = reference
        return items, dict(compiled=compiled, sampling=sampling,
            next_optimizer_update=update+1,
            timings=dict(compile_seconds=compile_seconds, rollout_seconds=rollout_seconds,
                         join_seconds=time.monotonic()-start),
            engine_passes=passes, independent_hf_passes=0,
            scope='Resident complete scheduled batch transport; no optimizer update, backend admission or efficacy claim.')
