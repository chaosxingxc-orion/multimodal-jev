"""Read the scoring reference independently of the Stage1 sampling strategy."""
import math


def scoring_settings(training, reference=None):
    if 'instance_negative_manifest' in training:
        if (not isinstance(reference, dict) or set(reference) != {'forward_rows', 'card_arm', 'zipf_exponent'}
                or type(reference['forward_rows']) is not int or reference['forward_rows'] < 1
                or not isinstance(reference['card_arm'], str) or not reference['card_arm']
                or type(reference['zipf_exponent']) not in (int, float)
                or not math.isfinite(reference['zipf_exponent']) or reference['zipf_exponent'] <= 0):
            raise ValueError('instance-negative training requires an explicit scoring reference')
        return reference
    if reference is not None:
        raise ValueError('this checkpoint binds scoring to the historical training settings')
    if 'class_scoring' in training:
        return training['class_scoring']
    return training['epoch_mining']
