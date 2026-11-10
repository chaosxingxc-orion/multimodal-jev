"""One-shot gated frozen actor dispatch; no queue, replay or weight updates."""
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

from multimodal_jev.omni_compact_acceptance import require_candidate

base = Path('/mnt/cephfs/xingchao/experiments/multimodal-jev')
root = base/'omni-compact-verl-train6-v1'
candidate = base/'omni-compact-keys-train6-v1'
preflight = base/'omni-compact-verl-preflight-v1'
prior = base/'omni-pretrained-forward-train6-v2'
old = base/'omni-pretrained-verl-train6-v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')
def inspect(cid):
    return json.loads(subprocess.check_output(['ssh', 'npu', shlex.join(['docker', 'inspect', cid])]))[0]

assert not root.exists(), 'one-shot root exists; inspect it, never automatically retry'
cx = inspect('742651b35fe69b6f7679a56136fd3643c03c07f915799937f70d6e6d87bd9da4')
assert cx['State']['Status'] == 'exited' and cx['State']['ExitCode'] == 0, 'candidate not successfully terminal'
with tempfile.TemporaryDirectory(prefix='jev-compact-admission-') as tmp:
    cp = Path(tmp)/'inspection.json'; write(cp, cx)
    accepted_candidate = require_candidate(candidate, cp, sha(cp))
cp_plan = read(candidate/'intent.json')
for name, pin in cp_plan['source_files'].items():
    assert hashlib.sha256(subprocess.check_output(['git', 'show', cp_plan['source_commit']+':'+name])).hexdigest() == pin

pl = read(preflight/'launch.json'); pp = read(preflight/'intent.json')
assert pl['container_id'] == '459db7b318cc73801b085727ac95dc04dc145543c11a47ddbd448b2ec749c2d5'
assert sha(preflight/'intent.json') == pl['intent_sha256'] == '27def22be94fa95659037efb1c4718577f336c858305629920b86127067e6cd8'
px = inspect(pl['container_id']); image = pp['image']; a = pp['argv']; host = px['HostConfig']
assert px['Image'] == image == cx['Image'] and px['State']['Status'] == 'exited' and px['State']['ExitCode'] == 0 and not px['State']['OOMKilled']
assert host['NanoCpus'] == 16*10**9 and host['Memory'] == 16*1024**3 and host['ShmSize'] == 2*1024**3
assert host['NetworkMode'] == 'none' and not host['Devices'] and not host['Privileged']
assert px['Config']['Cmd'] == a[a.index(image)+1:]
assert set(host['Binds']) == {a[i+1] for i,v in enumerate(a) if v == '-v'}
assert all(a[i+1] in px['Config']['Env'] for i,v in enumerate(a) if v == '-e')
assert read(preflight/'run/terminal.json') == {'exit_code': 0}
preflight_evidence = read(Path('docs/engineering/omni-compact-verl-preflight-v1-verified.json'))
assert sha(preflight/'run/result.json') == preflight_evidence['result_sha256'] == 'd683b4e55ae6779ac1c73f6a5bf587eefe8319d330286f443de3fc7a467f0665'
assert sha(preflight/'run/identity.json') == preflight_evidence['identity_sha256']
assert read(preflight/'run/result.json') == preflight_evidence['result']
assert pp['source_commit'] == pl['source_commit'] == 'dd1e4475c8c3b27befbed27d498c0c7e54150830'
for name, pin in pp['source_files'].items():
    assert sha(Path(pp['source_root'])/name) == pin
    assert hashlib.sha256(subprocess.check_output(['git', 'show', pp['source_commit']+':'+name])).hexdigest() == pin
assert pp['source_files']['src/multimodal_jev/omni_compact_attention.py'] == accepted_candidate['adapter_sha256']

acceptance = base/'omni-pretrained-forward-train6-v2-acceptance/acceptance.json'
assert sha(acceptance) == 'b9d34fcdb5069d723a5e9698b5445df815450db6fdd2debda971cc5111a91daa'
prior_launch = read(prior/'launch.json'); ox = inspect(prior_launch['container_id'])
assert ox['Id'] == read(acceptance)['container_id'] == 'e8f68cb9ebd5a607722e676eab977d08cf750fd2d86342b5001a3f3d3cc44be3'
assert ox['Image'] == image and ox['State']['Status'] == 'exited' and ox['State']['ExitCode'] == 0 and not ox['State']['OOMKilled']
assert sha(prior/'run/result.json') == read(acceptance)['result_sha256']
assert sha(prior/'intent.json') == read(acceptance)['intent_sha256']
pa = read(prior/'intent.json')['argv']; assert ox['Config']['Cmd'] == pa[pa.index(image)+1:]

root.mkdir()
write(root/'inspection.json', ox); write(root/'compact-inspection.json', cx)
assert sha(root/'compact-inspection.json') == accepted_candidate['inspection_sha256']
write(root/'candidate-acceptance.json', accepted_candidate)
write(root/'preflight-evidence.json', preflight_evidence)
for name, pin in pp['source_files'].items():
    target = root/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((Path(pp['source_root'])/name).read_bytes()); assert sha(target) == pin
plan = read(old/'intent.json')
argv = [s.replace(str(old), str(root)) for s in plan['argv']]
argv[argv.index('--name')+1] = root.name
argv[argv.index('--inspection-hash')+1] = sha(root/'inspection.json')
argv += ['--compact-keys-root', str(candidate), '--compact-keys-inspection', str(root/'compact-inspection.json'),
         '--compact-keys-inspection-hash', sha(root/'compact-inspection.json')]
assert '--preflight-only' not in argv
plan.update(argv=argv, source_commit=pp['source_commit'], source_files=pp['source_files'],
            inspection_sha256=sha(root/'inspection.json'), preflight_evidence_sha256=sha(root/'preflight-evidence.json'),
            compact_candidate=accepted_candidate, created_unix=time.time(),
            coordinator_sha256=sha(Path(__file__)),
            budget_reason='Original six frozen responses, actual verl microbatch sizes 1/2; unchanged four-hour budget and 0.1 gate. No optimizer or automatic replay.')
write(root/'intent.json', plan)
r = subprocess.run(['ssh', 'npu', shlex.join(argv)], text=True, capture_output=True)
write(root/'create-result.json', dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
assert r.returncode == 0, 'create failed; no retry'
cid = r.stdout.strip(); assert len(cid) == 64 and all(c in '0123456789abcdef' for c in cid), 'ambiguous create; inspect manually'
launch = dict(container_id=cid, image=image, source_commit=pp['source_commit'], intent_sha256=sha(root/'intent.json'))
write(root/'launch.json', launch)
r = subprocess.run(['ssh', 'npu', shlex.join(['docker', 'start', cid])], text=True, capture_output=True)
write(root/'start-result.json', dict(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr))
assert r.returncode == 0, 'start failed or ambiguous; no retry'
write(Path('docs/engineering/omni-compact-verl-train6-v1-launched.json'), dict(root=str(root), **launch))
print(json.dumps(dict(launch=launch, state=inspect(cid)['State'])))
