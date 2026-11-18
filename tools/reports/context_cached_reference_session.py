"""Feed sealed Stage1 results to the existing GRPO session, without ref forwards."""
from multimodal_jev.context_reference_dataset import ReferenceDataset
from tools.reports.context_dual_session_candidate import RuntimeReferenceCache


class TrainingReferenceDataset(ReferenceDataset, RuntimeReferenceCache):
    """Legacy session protocol, immutable storage implementation.

    The session checks misses before scoring behavior. read() raises on a miss,
    so its historical online-fill branch cannot execute with this dataset.
    """


def load(config, *, runtime_identity, policy_id, base_id, contract, candidate_ids):
    if set(config) != {'root','manifest_sha256','identity','population'}:
        raise ValueError('explicit reference dataset and full population required')
    identity=config['identity']
    if (identity['policy_id']!=policy_id or identity['base_id']!=base_id
            or identity['contract']!=contract or identity['runtime_identity']!=runtime_identity
            or identity['candidate_ids']!=list(candidate_ids)):
        raise ValueError('Stage1 reference, actual runtime and candidate ordering must match')
    return TrainingReferenceDataset(config['root'],manifest_sha256=config['manifest_sha256'],
        expected_identity=identity,expected_population=config['population'])
