"""Persisted normalization choice, independent of the reward composition."""


def normalization(config):
    value=config.get('norm_adv_by_std',True)
    if type(value) is not bool:raise ValueError('norm_adv_by_std must be boolean')
    return value


def validate_continuation(previous,current):
    if normalization(previous)!=normalization(current):
        raise ValueError('advantage normalization changed within trajectory')
