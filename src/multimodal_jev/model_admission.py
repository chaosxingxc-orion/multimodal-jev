"""Model configuration admission for deterministic actor comparisons."""
import math


def config_dropout_values(value, prefix='config'):
    """Collect dropout probabilities without assuming config dictionary key types."""
    result = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = prefix + '.' + str(key)
            if 'dropout' in str(key).lower() and isinstance(child, (int, float)):
                if not math.isfinite(child) or not 0 <= child <= 1:
                    raise ValueError('invalid dropout probability: ' + name)
                result[name] = child
            result.update(config_dropout_values(child, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(config_dropout_values(child, prefix + '.' + str(index)))
    return result
