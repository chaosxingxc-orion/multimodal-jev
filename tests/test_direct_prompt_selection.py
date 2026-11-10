import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('direct_select',
    Path(__file__).parents[1]/'tools/jev/select_direct_prompt.py')
# CLI sibling module is intentionally imported as a normal script dependency.
import sys
sys.path.insert(0, str(Path(__file__).parents[1]/'tools/jev'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(name, all_six, pairs, total, length=100):
    return {'id': name, 'instruction_characters': length,
            'metrics': {'all_six_correct': all_six,
                        'pairs': {'p': {'both_correct': pairs}},
                        'by_variant': {'v': {'correct': total}}}}


def test_paired_discrimination_precedes_marginal_and_parent_wins_tie():
    parent = row('parent', 0, 0, 1000)
    discriminating = row('p1', 0, 1, 100)
    assert min([parent, discriminating], key=module.rank)['id'] == 'p1'
    assert min([parent, row('p2', 0, 0, 1000, 1)], key=module.rank)['id'] == 'parent'
    assert min([row('p1', 1, 4, 6), row('p2', 0, 200, 1000)], key=module.rank)['id'] == 'p1'
    assert min([row('p2', 0, 1, 100, 90), row('p1', 0, 1, 100)], key=module.rank)['id'] == 'p2'
