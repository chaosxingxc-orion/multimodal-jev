#!/usr/bin/env python3
"""Build a self-contained listening packet; never grant dataset admission."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from verify_intent_development_review import verify


def render(rows,packet_sha256,audit_sha256):
    data=json.dumps(dict(packet_sha256=packet_sha256,audit_sha256=audit_sha256,records=rows),ensure_ascii=False)
    # Script data must not allow a reference string to terminate the element.
    data=data.replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    return '''<!doctype html><meta charset="utf-8"><title>SLURP 音频审核</title>
<style>body{max-width:900px;margin:32px auto;font:17px system-ui;line-height:1.6;padding:16px}button,select,input,textarea{font:inherit;margin:8px 4px}textarea{width:95%;height:100px}audio{width:100%}.muted{color:#555}#reference{font-size:23px}label{display:block}button{padding:8px}</style>
<h1>SLURP 开发候选音频审核</h1><p>核对听到的内容与参考文本，再独立判断意图。模糊项保留待定；不要推断未说出的对象或动作。本页面不自动授予训练、评测或模型选择资格。</p>
<label>审核人 <input id="reviewer" placeholder="填写姓名或稳定标识"></label>
<button id="prev">上一条</button><select id="index"></select><button id="next">下一条</button><span id="progress"></span>
<p id="identity" class="muted"></p><audio id="audio" controls preload="none"></audio><p id="reference"></p><p id="intent"></p>
<label>音频与参考：<select id="alignment"><option value="pending">待审核／无法确定</option><option value="matches">内容相符</option><option value="mismatch">内容不符</option><option value="unintelligible">无法辨认</option></select></label>
<label>意图判断：<select id="ontology"><option value="pending">待审核／模糊</option><option value="consistent">与音频语义相符</option><option value="conflict">标签冲突</option></select></label>
<label>判断依据与问题记录<textarea id="notes"></textarea></label><button id="save">记录本条</button><button id="export">导出审核 JSON</button><p id="message" role="status"></p>
<p class="muted">请及时导出保存。浏览器本地缓存仅用于恢复未完成工作；音频在本页内播放，不上传。</p>
<script id="packet" type="application/json">'''+data+'''</script><script>
const packet=JSON.parse(document.getElementById('packet').textContent),$=id=>document.getElementById(id),key='slurp-review-'+packet.packet_sha256;let state={reviewer:'',decisions:{}},current=0,dirty=false;
try{const saved=JSON.parse(localStorage.getItem(key)||'null');if(saved)state=saved;}catch(e){}
$('reviewer').value=state.reviewer||'';
packet.records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' / '+packet.records.length;$('index').append(o);});
function persist(){state.reviewer=$('reviewer').value.trim();try{localStorage.setItem(key,JSON.stringify(state));}catch(e){$('message').textContent='浏览器缓存不可用，请导出保存。';}}
function show(i){current=i;dirty=false;const r=packet.records[i],s=state.decisions[r.id]||{};$('index').value=i;$('audio').src=r.audio_data;$('reference').textContent=r.reference;$('intent').textContent='原标签：'+r.intent;$('identity').textContent=r.id;$('alignment').value=s.alignment||'pending';$('ontology').value=s.ontology||'pending';$('notes').value=s.notes||'';$('progress').textContent='已记录 '+Object.keys(state.decisions).length+' / '+packet.records.length;}
function save(){if(!$('reviewer').value.trim()||!$('notes').value.trim()){$('message').textContent='请填写审核人和判断依据；当前修改尚未保存。';return false;}const r=packet.records[current];state.decisions[r.id]={id:r.id,source_group:r.source_group,audio_sha256:r.audio_sha256,record_sha256:r.record_sha256,alignment:$('alignment').value,ontology:$('ontology').value,notes:$('notes').value.trim(),reviewer:$('reviewer').value.trim(),recorded_at:new Date().toISOString(),evaluation_allowed:false,optimization_allowed:false};persist();dirty=false;$('message').textContent='已记录。';$('progress').textContent='已记录 '+Object.keys(state.decisions).length+' / '+packet.records.length;return true;}
function flush(){return !dirty||save();}
function navigate(i){if(flush())show(i);else $('index').value=current;}
for(const id of ['alignment','ontology','notes']){$(id).oninput=()=>{dirty=true;};$(id).onchange=()=>{dirty=true;};}
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
$('save').onclick=save;$('index').onchange=()=>navigate(Number($('index').value));$('prev').onclick=()=>navigate(Math.max(0,current-1));$('next').onclick=()=>navigate(Math.min(packet.records.length-1,current+1));$('reviewer').onchange=persist;
$('export').onclick=()=>{if(!flush())return;persist();const result={schema:'slurp-intent-listening-review-v1',packet_sha256:packet.packet_sha256,reference_audit_sha256:packet.audit_sha256,reviewer:state.reviewer,records:packet.records.map(r=>state.decisions[r.id]||{id:r.id,source_group:r.source_group,audio_sha256:r.audio_sha256,record_sha256:r.record_sha256,alignment:'pending',ontology:'pending',notes:'',evaluation_allowed:false,optimization_allowed:false}),evaluation_allowed:false,optimization_allowed:false};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='slurp-intent-listening-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};show(0);
</script>'''


def build(packet_path,audit_path,output):
    import soundfile as sf
    from multimodal_jev.slurp_training import content_hash
    packet_path,audit_path,output=map(Path,(packet_path,audit_path,output))
    if output.exists():raise FileExistsError('review packet already exists')
    verify(packet_path,audit_path)
    packet=json.loads(packet_path.read_text());rows=[]
    for source in packet:
        r=source['request'];path=Path(r['input']['audio'])
        if file_hash(path)!=r['audio_sha256']:raise ValueError('audio changed')
        audio,sr=sf.read(path,dtype='int16')
        if sr!=16000 or audio.ndim!=1:raise ValueError('mono16k audio required')
        wav=io.BytesIO();sf.write(wav,audio,sr,format='WAV',subtype='PCM_16')
        t=source['target']['native_target']
        rows.append(dict(id=r['id'],source_group=r['source_group'],audio_sha256=r['audio_sha256'],record_sha256=content_hash(source),
            reference=source['reference'],intent=t['scenario']+'_'+t['action'],audio_data='data:audio/wav;base64,'+base64.b64encode(wav.getvalue()).decode()))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(render(rows,file_hash(packet_path),file_hash(audit_path)))
    return dict(output=str(output),sha256=file_hash(output),bytes=output.stat().st_size,recordings=len(rows),
        packet_sha256=file_hash(packet_path),audit_sha256=file_hash(audit_path),implementation_sha256=file_hash(__file__),
        audio_rendering='16k mono PCM16 WAV preview from original FLAC; original audio hash retained',
        evaluation_allowed=False,optimization_allowed=False,scope='Listening UI only. No listening or semantic adjudication performed by this builder.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True)
    a=p.parse_args();result=build(a.packet,a.audit,a.output);atomic_json(Path(a.receipt),result);print(json.dumps(result,indent=2))
