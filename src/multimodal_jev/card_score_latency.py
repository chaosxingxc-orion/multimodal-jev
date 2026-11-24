"""Host-observable first scores from a frozen ordered block, in monotonic time."""
import time


class ScoreLatency:
    def __init__(self, candidates, clock=time.monotonic):
        if type(candidates) is not int or candidates <= 0:
            raise ValueError('positive candidate width required')
        self.width = candidates
        self.clock = clock
        self.start = clock()
        self.first_score = self.first_recording = None
        self.pending = {}

    def visible(self, rows):
        # Caller has already materialized the probabilities on the host.
        if self.first_recording is not None:
            return
        now = self.clock() - self.start
        if self.first_score is None:
            self.first_score = now
        for row in rows:
            uid, column = row['request_id'].rsplit(':', 1)
            found = self.pending.setdefault(uid, set())
            index = int(column)
            if not 0 <= index < self.width or index in found:
                raise ValueError('invalid first-recording candidate identity')
            found.add(index)
            if len(found) == self.width:
                self.first_recording = now
                self.pending.clear()
                break

    def result(self):
        if self.first_recording is None:
            raise ValueError('no complete recording became host-visible')
        return dict(first_score_seconds=self.first_score,
                    first_decision_seconds=self.first_recording,
                    scoring_seconds=self.clock() - self.start)
