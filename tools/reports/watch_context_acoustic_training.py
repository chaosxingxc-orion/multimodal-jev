"""Reuse the fixed 900-second watcher with acoustic-specific launch validation."""
import argparse
from types import FunctionType

from multimodal_jev.local_config import load_config
from tools.reports import context_acoustic_training_operations as acoustic
from tools.reports import observe_context_macro_training as observer
from tools.reports import watch_context_macro_training as watcher


def bind(function, **overrides):
    # Independent globals: the original observer remains unchanged for other jobs.
    result = FunctionType(function.__code__, dict(function.__globals__, **overrides),
                          function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


observe = bind(observer.observe, validate=acoustic.validate, command=acoustic.command)
run = bind(watcher.run, observe=observe)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    run(c, provenance)
