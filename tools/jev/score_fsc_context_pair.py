"""Bind target scoring to independently re-audited inference and frozen inputs."""
import argparse
import json
from pathlib import Path

from tools.jev.collect_fsc_context_pair import audit
from tools.jev.context_rl_evaluation_plan import sha
from tools.jev.score_fsc_context_transfer import run as score


def read(path):return json.loads(Path(path).read_text())


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def run(spec, output):
    """All identities and inference are checked before scorer opens target CSV."""
    output = Path(output)
    if output.exists():raise ValueError('new score output required')
    # The caller seals this contract before source packaging / target inference.
    contract = spec['scoring_contract']
    if sha(contract['path']) != contract['sha256']:
        raise ValueError('predeclared scoring contract changed')
    frozen = read(contract['path'])
    for entry in frozen.values():
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('frozen target definitions or inputs changed')
    args = {k:spec[k] for k in ('package','digest','collection')}
    for name in ('launch','observed','transport'):args[name] = read(spec[name])
    inference = audit(**args)
    previous = read(spec['inference_report'])
    if any(previous.get(k) != v for k,v in inference.items()):
        raise ValueError('persisted inference proof differs from actual artifacts')
    inspected = previous['runtime_inspect']
    if (Path(inspected['path']).resolve() != Path(spec['observed']).resolve()
            or inspected['sha256'] != sha(spec['observed'])):
        raise ValueError('runtime inspection binding differs')
    package = Path(spec['package'])
    source = read(package/'frozen-source/PACKAGE.json')
    manifest = read(package/'inputs/MANIFEST.json')
    expected = dict(task=manifest['task_sha256'], population=manifest['population_sha256'],
        input_package_manifest=sha(package/'inputs/MANIFEST.json'),
        input_audit=sha(package/'inputs/audit.json'),
        input_receipts=sha(package/'inputs/receipts.jsonl'))
    if set(frozen) != set(expected) or any(frozen[k]['sha256'] != h for k,h in expected.items()):
        raise ValueError('scoring ontology/population differs from executed inputs')
    policies = {role:dict(policy_id=source['policies'][role]['policy_id'],
                         predictions=inference['predictions'][role])
                for role in ('initial','selected')}
    scoring = {k:frozen[k] for k in ('population','task','input_audit','input_package_manifest')}
    scoring.update(source_selection=artifact(package/'frozen-source/source-selection.json'),
                   input_receipts=frozen['input_receipts']['path'], policies=policies)
    manifest_path = output.with_suffix('.manifest.json')
    with manifest_path.open('x') as stream:json.dump(scoring,stream,indent=2)
    result = score(manifest_path)
    result.update(runtime_provenance_verified=True,
        inference_proof=artifact(spec['inference_report']), runtime_inspect=artifact(spec['observed']),
        scoring_contract=contract, source_manifest_sha256=inference['source_manifest_sha256'],
        scope='Full FSC test scoring after independent source-selected policy/runtime/input audit. '
              'Frozen native-task configuration transfer; no target tuning, semantic-presence '
              'transfer, statistical significance or RL-only causal attribution claimed.')
    with output.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(read(args.spec),args.output)
    print(json.dumps(dict(runtime_provenance_verified=result['runtime_provenance_verified'],
                          comparison=result['comparison']),indent=2))
