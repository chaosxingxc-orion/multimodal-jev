#!/usr/bin/env python3
"""Check structural and known textual contradictions, not semantic certification."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(package, definitions, policies, boundaries):
    issues = []
    counts = {}
    role_versions = defaultdict(lambda: defaultdict(list))
    names = [r['entity_type'] for r in definitions['types']]
    policy_names = [r['entity_type'] for r in policies['policies']]
    if len(names) != len(set(names)) or len(policy_names) != len(set(policy_names)):
        issues.append(dict(code='duplicate_type_or_policy'))
    if set(names) != set(policy_names):
        issues.append(dict(code='definition_policy_membership_mismatch'))
    scope = package['scope']
    if definitions.get('scope', scope) != scope:
        issues.append(dict(code='document_scope_mismatch', path='definitions/scope'))
    for label, doc in [('definitions', definitions), ('policy', policies), ('boundaries', boundaries)]:
        if doc.get('scope_contract') != package.get('scope_contract'):
            issues.append(dict(code='scope_contract_pointer_mismatch', path=label))
    pairs = set()
    for row in boundaries['rules']:
        pair = tuple(sorted(row['types']))
        if len(pair) != 2 or len(set(pair)) != 2 or not set(pair) <= set(names):
            issues.append(dict(code='unknown_or_invalid_boundary_pair', types=row['types']))
        if pair in pairs:
            issues.append(dict(code='duplicate_boundary_pair', types=row['types']))
        pairs.add(pair)
    for label, doc, collection, group in [
        ('definitions', definitions, 'types', 'type'),
        ('policy', policies, 'policies', 'policy'),
        ('boundaries', boundaries, 'rules', 'boundary'),
    ]:
        counts[label] = len(doc[collection])
        for row in doc[collection]:
            identity = row.get('entity_type', '/'.join(row.get('types', [])))
            path = label + '/' + identity
            if label == 'definitions' and row.get('scope') != scope:
                issues.append(dict(code='type_scope_mismatch', path=path))
            active = {k: row[k] for k in package['runtime_field_groups'][group] if k in row}
            rendered = json.dumps(active, ensure_ascii=False)
            if scope == 'expressed_communicative_content' and re.search(
                r'request.relevan|unrelated background (?:alone does not count|is outside)', rendered, re.I
            ):
                issues.append(dict(code='stale_request_scope', path=path))
            # Specific known bad rule; this is deliberately not a natural-language theorem prover.
            if 'If the current input does not establish the role required by either definition, preserve that predicate as unresolved.' in rendered:
                issues.append(dict(code='absence_conflated_with_unavailable', path=path))
            if 'including descriptive words within a playlist name under an admitted span policy' in rendered:
                issues.append(dict(code='title_word_role_shortcut', path=path))
            seen = set()
            for rule in row.get('role_rules', []):
                rid = rule['id']
                if rid in seen:
                    issues.append(dict(code='duplicate_role_rule_id', path=path, rule_id=rid))
                seen.add(rid)
                role_versions[rid][json.dumps(rule, sort_keys=True, ensure_ascii=False)].append(path)
    for rid, variants in role_versions.items():
        if len(variants) > 1:
            issues.append(dict(code='shared_role_rule_drift', rule_id=rid,
                               variant_paths=list(variants.values())))
    return dict(status='pass' if not issues else 'fail', issues=issues, counts=counts,
                shared_role_ids=len(role_versions), semantic_certification=False,
                limitation='Finite structural and exact-pattern checks; pass does not prove all natural-language rules consistent or labels correct.')


def audit_file(package_path):
    package = read(package_path)
    documents = [read(package[k]) for k in ('definitions', 'policy', 'boundaries')]
    result = audit(package, *documents)
    d, p, b = documents
    for field, actual, expected in [
        ('policy/source_definition_sha256', p['source_definition_sha256'], sha(package['definitions'])),
        ('boundaries/source_definition_sha256', b['source_definition_sha256'], sha(package['definitions'])),
        ('boundaries/source_policy_sha256', b['source_policy_sha256'], sha(package['policy'])),
    ]:
        if actual != expected:
            result['issues'].append(dict(code='document_pin_mismatch', path=field))
    for label, doc in [('package', package), ('definitions', d), ('policy', p), ('boundaries', b)]:
        if doc.get('scope_contract'):
            if sha(doc['scope_contract']) != doc.get('scope_contract_sha256'):
                result['issues'].append(dict(code='scope_contract_pin_mismatch', path=label))
    result.update(status='pass' if not result['issues'] else 'fail', package=str(package_path),
                  package_sha256=sha(package_path), implementation_sha256=sha(__file__),
                  files={package[k]: sha(package[k]) for k in ('definitions', 'policy', 'boundaries')},
                  training_dispatch_allowed=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_file(args.package)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'issues')}, indent=2))
    print('issues:', len(result['issues']))
    raise SystemExit(0 if result['status'] == 'pass' else 1)
