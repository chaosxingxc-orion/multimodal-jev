import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_intent_audio_review import render


def test_reference_cannot_inject_script_and_packet_is_roundtrippable():
    rows=[{'reference':'</script><script>alert(1)</script>&','id':'sample'}]
    page=render(rows,'packet-hash','audit-hash')
    embedded=re.search(r'<script id="packet" type="application/json">(.*?)</script>',page,re.S).group(1)
    assert '<' not in embedded and json.loads(embedded)['records']==rows
    assert 'evaluation_allowed:false' in page and 'optimization_allowed:false' in page
    assert 'textContent=r.reference' in page
