import pytest

from tools.jev.collect_context_replica_training import bindings, transport_names


def launch(names, directories=('run', 'supervision', 'reference-cache')):
    return dict(manifest_sha256='prepared-hash', config_sha256='config-hash',
                services={name: f'{name}-hash' for name in names},
                layout=dict(configs=['config.json', 'run.py', 'RUNTIME-SOURCES.json',
                                     'PREPARED.json'],
                            services={name: f'service-{name}.json' for name in names},
                            directories=[*directories, *[f'service-{n}' for n in names]]))


@pytest.mark.parametrize('names', [['a'], ['a', 'b'], ['a', 'b', 'c']])
def test_transport_set_is_read_from_declared_layout(names):
    names_seen, directories, layout = transport_names(launch(names))
    assert names_seen == ['config.json', 'run.py', 'RUNTIME-SOURCES.json', 'PREPARED.json',
                          *[f'service-{n}.json' for n in names]]
    assert directories == ['run', 'supervision', 'reference-cache',
                           *[f'service-{n}' for n in names]]


@pytest.mark.parametrize('names', [['a'], ['a', 'b']])
def test_launch_bindings_cover_every_declared_file(names):
    record = launch(names)
    _, _, layout = transport_names(record)
    assert bindings(record, layout) == dict(
        **{'config.json': 'config-hash', 'PREPARED.json': 'prepared-hash'},
        **{f'service-{n}.json': f'{n}-hash' for n in names})
