import json
from pathlib import Path
import re
import sys
import shutil
import subprocess
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_intent_audio_review import render


def test_reference_cannot_inject_script_and_packet_is_roundtrippable():
    rows=[{'reference':'</script><script>alert(1)</script>&','id':'sample'}]
    page=render(rows,'packet-hash','audit-hash')
    embedded=re.search(r'<script id="packet" type="application/json">(.*?)</script>',page,re.S).group(1)
    assert '<' not in embedded and json.loads(embedded)['records']==rows
    assert 'evaluation_allowed:false' in page and 'optimization_allowed:false' in page
    assert 'textContent=r.reference' in page


def test_unsaved_edits_survive_navigation_and_export(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node required for actual review JavaScript execution')
    rows=[dict(id=k,source_group=k,audio_sha256='audio',record_sha256='record',reference='words',intent='alarm_set',audio_data='audio') for k in ('a','b')]
    page=render(rows,'packet','audit')
    packet,script=re.findall(r'<script[^>]*>(.*?)</script>',page,re.S)
    harness=r'''
const vm=require('vm'),assert=require('assert');
const elements={},handlers={},stored={};let blobs=[];
const element=id=>elements[id]||(elements[id]={value:'',textContent:'',append(){},click(){}});
element('packet').textContent=PACKET;
const context={document:{getElementById:element,createElement:()=>({append(){},click(){}})},
 window:{addEventListener:(name,fn)=>handlers[name]=fn},
 localStorage:{getItem:k=>stored[k]||null,setItem:(k,v)=>stored[k]=v},
 URL:{createObjectURL:b=>{blobs.push(JSON.parse(b.parts[0]));return 'blob:test';},revokeObjectURL(){}},
 Blob:class{constructor(parts){this.parts=parts;}},setTimeout:fn=>fn()};
vm.createContext(context);vm.runInContext(SCRIPT,context);
element('alignment').value='matches';element('alignment').onchange();
element('next').onclick();assert.equal(element('identity').textContent,'a');
element('export').onclick();assert.equal(blobs.length,0);
element('index').value=1;element('index').onchange();assert.equal(element('index').value,0);
let prevented=false;handlers.beforeunload({preventDefault(){prevented=true;}});assert(prevented);
element('reviewer').value='human';element('notes').value='listened';element('notes').oninput();
element('ontology').value='consistent';element('ontology').onchange();
element('next').onclick();assert.equal(element('identity').textContent,'b');
element('prev').onclick();assert.equal(element('notes').value,'listened');
element('notes').value='rechecked';element('notes').oninput();element('export').onclick();
assert.equal(blobs.length,1);assert.equal(blobs[0].records[0].notes,'rechecked');
assert.equal(blobs[0].records[0].reviewer,'human');assert.equal(blobs[0].records[1].alignment,'pending');
assert.equal(blobs[0].evaluation_allowed,false);assert.equal(blobs[0].optimization_allowed,false);
prevented=false;handlers.beforeunload({preventDefault(){prevented=true;}});assert(!prevented);
'''
    path=tmp_path/'review.js';path.write_text('const PACKET='+json.dumps(packet)+';const SCRIPT='+json.dumps(script)+';'+harness)
    subprocess.run([node,str(path)],check=True,capture_output=True,text=True)
