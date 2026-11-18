"""Opt-in acoustic batching in the unchanged authenticated service lifecycle."""
from types import FunctionType

from tools.reports import context_dual_runtime_candidate as parent
from tools.reports.context_acoustic_owner_candidate import owner_class


def service_namespace(config):
    if config['policy'].get('process_device_scope') is not None:
        raise ValueError('isolated acoustic TP1 service required')
    namespace = parent.service_namespace(config)
    namespace['owner_class'] = lambda root: owner_class(root, acoustic=True)
    return namespace


# Keep all socket, policy activation, error handling and teardown behavior.
main = FunctionType(parent.main.__code__, dict(parent.main.__globals__,
    service_namespace=service_namespace), parent.main.__name__, parent.main.__defaults__,
    parent.main.__closure__)


if __name__ == '__main__':
    main()
