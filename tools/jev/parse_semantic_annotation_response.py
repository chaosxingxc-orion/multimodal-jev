"""Strict offline annotation envelope parsing; separate from benchmark scoring."""
import json
import re

from consolidate_intent_presence_references import require


def parse_response_content(content):
    """Accept JSON or one complete JSON fence; never extract a guessed substring.

    Preserve the representation in audit results. This is an offline annotation
    transport parser, not the native model benchmark parser.
    """
    require(isinstance(content, str), 'response content must be text')
    text = content.strip()
    representation = 'bare_json'
    if text.startswith('```'):
        match = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```', text)
        require(match is not None, 'not a single complete JSON fence')
        text = match.group(1)
        representation = 'json_fence'
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    return value, representation

