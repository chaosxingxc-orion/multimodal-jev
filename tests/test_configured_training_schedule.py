from copy import deepcopy
import json

import pytest

from tools.prepare_training_schedule import PROFILE, validate


def config():
    return dict(schema='configured-context-training-schedule-v1', release='release',
        release_sha256='identity', ledger='ledger', ledger_sha256='identity',
        profile=deepcopy(PROFILE), output_directory='output')


def test_fixed_profile_accepted():
    validate(config())


@pytest.mark.parametrize('key,value', [('seed', 318), ('epochs', 1), ('global_batch', 256),
    ('microbatch', 32), ('world_size', 8), ('split', 'selection'),
    ('oversampling', True), ('padding_weight', False), ('length_grouping', 'global')])
def test_unsupported_knobs_not_silently_ignored(key, value):
    value_config = config()
    value_config['profile'][key] = value
    with pytest.raises(ValueError, match='Unsupported schedule profile'):
        validate(value_config)


def test_extra_fields_rejected():
    value = config(); value['sample_count'] = 100
    with pytest.raises(ValueError):
        validate(value)
