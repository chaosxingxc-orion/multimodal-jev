"""Resident finite-actor lifecycle with collective policy acceptance boundaries.

The caller owns model/optimizer loading, checkpoint publication and independent
backend verification. This component never turns a policy name into acceptance.
All ranks must enter every method together, including failure handling.
"""

class SpanTrainingSession:
    def __init__(self, actor, *, policy_id, completed_updates, gather):
        if not isinstance(policy_id, str) or not policy_id or type(completed_updates) is not int or completed_updates < 0:
            raise ValueError('accepted initial policy and nonnegative cursor required')
        if actor.config.jev_policy_id != policy_id:
            raise ValueError('loaded actor policy differs')
        self.actor = actor
        self.policy_id = policy_id
        self.completed_updates = completed_updates
        self.gather = gather
        self.phase = 'accepted'
        self.pending_policy = None

    def _agree(self, operation, expected_phase, error=None):
        local = dict(operation=operation, phase=self.phase, policy_id=self.policy_id,
                     completed_updates=self.completed_updates, pending_policy=self.pending_policy,
                     error=error or (None if self.phase == expected_phase else 'invalid session phase'))
        peers = self.gather(local)
        if not peers or any(p != local for p in peers) or any(p['error'] for p in peers):
            raise ValueError('collective session transition rejected: '+str(peers))

    def update(self, batch):
        self._agree('update', 'accepted', None if self.actor.config.jev_policy_id == self.policy_id else 'actor identity changed')
        # An exception may occur after optimizer mutation. No automatic retry or
        # accepted-policy rollback without restoring exact model/Adam together.
        self.phase = 'updating'
        try:
            metrics = self.actor.update_policy(batch)
        except BaseException:
            self.phase = 'failed'
            raise
        self.completed_updates += 1
        self.phase = 'awaiting_publication'
        return metrics

    def publish(self, policy_id):
        error = None if isinstance(policy_id, str) and policy_id else 'nonempty exported policy identity required'
        self._agree('publish:'+str(policy_id), 'awaiting_publication', error)
        self.pending_policy = policy_id
        self.phase = 'awaiting_acceptance'

    def accept(self, verify):
        """verify(policy_id, updates) authenticates the actual independent gate.

        Its returned receipt must include policy_id/updates/passed; caller must
        verify files/exits/probabilities rather than trust those fields alone.
        """
        self._agree('accept', 'awaiting_acceptance')
        error = None; receipt = None
        try:
            receipt = verify(self.pending_policy, self.completed_updates)
            if (receipt['passed'] is not True or receipt['policy_id'] != self.pending_policy
                    or receipt['updates'] != self.completed_updates):
                raise ValueError('independent acceptance identity or cursor differs')
        except Exception as exc:
            error = repr(exc)
        try:
            self._agree('accept_result', 'awaiting_acceptance', error)
        except BaseException:
            self.phase = 'failed'
            raise
        self.policy_id = self.pending_policy
        self.pending_policy = None
        self.actor.config.jev_policy_id = self.policy_id
        self.phase = 'accepted'
        return receipt

    def checkpoint_cursor(self, *, schedule_sha256, checkpoint_id):
        """Serialize only accepted boundaries; caller binds real state contents."""
        error = None if all(isinstance(x, str) and x for x in (schedule_sha256, checkpoint_id)) else 'checkpoint/schedule identities required'
        self._agree('checkpoint:'+str(schedule_sha256)+':'+str(checkpoint_id), 'accepted', error)
        return dict(schema='jev-span-training-session-v1', completed_updates=self.completed_updates,
                    policy_id=self.policy_id, schedule_sha256=schedule_sha256, checkpoint_id=checkpoint_id)

    @classmethod
    def restore(cls, actor, cursor, *, schedule_sha256, checkpoint_id, gather):
        if (cursor['schema'] != 'jev-span-training-session-v1' or cursor['schedule_sha256'] != schedule_sha256
                or cursor['checkpoint_id'] != checkpoint_id):
            raise ValueError('session checkpoint or schedule differs')
        return cls(actor, policy_id=cursor['policy_id'], completed_updates=cursor['completed_updates'], gather=gather)
