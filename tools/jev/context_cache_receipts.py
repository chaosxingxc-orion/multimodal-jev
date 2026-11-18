"""Parse an audit stream with one known Ascend shutdown diagnostic."""
import json

DRIVER_TAIL='DrvMngGetConsoleLogLevel failed. (ret=4)'


def parse_stream(text):
    lines=text.splitlines();diagnostics=[]
    if lines and lines[-1]==DRIVER_TAIL:
        diagnostics=[lines.pop()]
    records=[json.loads(line) for line in lines]
    if (len(records)<3 or records[0].get('kind')!='terminal' or
            records[-1].get('kind')!='manifest' or
            any(r.get('kind')!='input' for r in records[1:-1])):
        raise ValueError('one terminal, inputs and final manifest required')
    if records[0]['value']['exit_code']!=0:raise ValueError('failed audit terminal')
    if len(records)-2!=records[-1]['value']['recordings']:raise ValueError('incomplete audit stream')
    return records,diagnostics
