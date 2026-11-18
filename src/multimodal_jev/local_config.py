"""Strict JSON macro resolution; machine bindings are read only from a local file.

No shell evaluation, environment interpolation, implicit path guessing or secret
lookup. A whole-value macro preserves JSON types; embedded macros require text.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


TOKEN = re.compile(r'\$\{([A-Z][A-Z0-9_]*)\}')
NAME = re.compile(r'[A-Z][A-Z0-9_]*\Z')


def resolve(config, macros):
    if not isinstance(macros, dict) or any(not NAME.fullmatch(k) for k in macros):
        raise ValueError('Macro names must match [A-Z][A-Z0-9_]*')
    cached, used = {}, set()

    def macro(name, stack):
        used.add(name)
        if name not in macros:
            raise ValueError('Missing local macro: ' + name)
        if name in stack:
            raise ValueError('Macro cycle: ' + ' -> '.join((*stack, name)))
        if name not in cached:
            cached[name] = visit(macros[name], (*stack, name))
        return deepcopy(cached[name])

    def visit(value, stack=()):
        if isinstance(value, dict):
            if any('${' in key for key in value):
                raise ValueError('Macros in object keys are unsupported')
            return {key: visit(child, stack) for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, stack) for child in value]
        if not isinstance(value, str):
            return value
        match = TOKEN.fullmatch(value)
        if match:
            return macro(match[1], stack)
        def replace(match):
            result = macro(match[1], stack)
            if not isinstance(result, str):
                raise ValueError('Embedded macro must be a string: ' + match[1])
            return result
        result = TOKEN.sub(replace, value)
        if '${' in result:
            raise ValueError('Malformed or unresolved macro expression')
        return result

    result = visit(config)
    return result, sorted(used)


def load_config(template, local):
    """Return resolved config and provenance without exposing macro values."""
    template, local = Path(template), Path(local)
    template_bytes, local_bytes = template.read_bytes(), local.read_bytes()
    bindings = json.loads(local_bytes)
    if set(bindings) != {'schema', 'macros'} or bindings['schema'] != 'jev-local-macros-v1':
        raise ValueError('Expected jev-local-macros-v1 with macros')
    resolved, used = resolve(json.loads(template_bytes), bindings['macros'])
    encoded = json.dumps(resolved, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return resolved, dict(schema='jev-resolved-config-v1',
        template_sha256=hashlib.sha256(template_bytes).hexdigest(),
        local_bindings_sha256=hashlib.sha256(local_bytes).hexdigest(),
        resolved_semantic_sha256=hashlib.sha256(encoded).hexdigest(), macros_used=used)
