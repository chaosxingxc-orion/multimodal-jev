"""KV training binding for the shared fixed 900-second observer."""
import argparse
from multimodal_jev.local_config import load_config
from tools.reports import context_kv_training_operations as training
from tools.reports import observe_context_macro_training as observer
from tools.reports import watch_context_macro_training as watcher
from tools.reports.watch_context_acoustic_training import bind

observe = bind(observer.observe, validate=training.validate, command=training.command)
run = bind(watcher.run, observe=observe)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    run(c, provenance)
