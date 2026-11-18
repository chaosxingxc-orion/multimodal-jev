"""Finite tail check using the unchanged cached-reference GRPO update path.

Not a released launcher: source packaging, parent/reference/parity admission,
independent output audit and actual NPU execution remain required.
"""
import ast
from pathlib import Path

from multimodal_jev.context_cached_tail_window import contract
from tools.reports.context_cached_reference_trainer import generate as cached_generate


def generate(source_root):
    source = cached_generate(source_root)
    replacements = [
        ("        from tools.reports.context_macro_trainer_candidate import experiment_window\n"
         "        experiment_window(c)",
         "        from multimodal_jev.context_cached_tail_window import contract as tail_contract,select as select_tail\n"
         "        tail=tail_contract(c)"),
        ("        progress('load')", "        contract['tail_admission']=tail\n        progress('load')"),
        ("                if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:raise ValueError('full learning population required')",
         "                if len(population)!=c['batch_config']['source_queries'] or len(batches)!=layout['total_updates']:raise ValueError('full learning population required')\n"
         "                batches=select_tail(batches,c,population.keys())"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('cached GRPO tail integration anchor changed')
        source = source.replace(before, after)
    ast.parse(source)
    return source


def run(config, output):
    contract(config)
    namespace = {'__name__': __name__ + '.derived'}
    exec(compile(generate(config['source_root']), '<cached-grpo-tail>', 'exec'), namespace)
    return namespace['run'](config, Path(output))


if __name__ == '__main__':
    import argparse
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    contract(config)
    args.output.mkdir(parents=True, exist_ok=True)
    run(config, args.output)
