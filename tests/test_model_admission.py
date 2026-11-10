import pytest
from multimodal_jev.model_admission import config_dropout_values


def test_integer_label_keys_do_not_hide_nested_dropout():
    config = {'id2label': {0: 'no', 1: 'yes'}, 'layers': [{2: {'attention_dropout': 0.1}}],
              'text_config': {'attention_dropout': 0.0}, 'dropout_description': 'disabled'}
    assert config_dropout_values(config) == {
        'config.layers.0.2.attention_dropout': 0.1, 'config.text_config.attention_dropout': 0.0}


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -0.1, 1.1])
def test_invalid_dropout_fails_admission(value):
    with pytest.raises(ValueError, match='invalid dropout'):
        config_dropout_values({'attention_dropout': value})
