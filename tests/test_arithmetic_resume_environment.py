import ast
import json

import pytest

from tools.reports.check_arithmetic_resume_environment import validate,RUNNER
from tools.reports.prepare_context_dual_throughput_pair import sha


def fixture(tmp_path):
    package=tmp_path/'package';package.mkdir()
    c=dict(resume_origin_config='/original/config.json',encoded_audio_cache='/features/run/cache',
        audio_mirror_root='/audio/run',pins={'/original/config.json':'x','/shared/model/weights':'y'})
    (package/'config.json').write_text(json.dumps(c))
    manifest=dict(status='arithmetic_recovery_prefix_prepared_only',training_dispatch_allowed=False,
        finite_dispatch_allowed=False,remote_root='/candidate',files={'config.json':sha(package/'config.json')})
    (package/'PREPARED.json').write_text(json.dumps(manifest))
    return dict(package=str(package),package_sha256=sha(package/'PREPARED.json'),tag='cpu-check',host='npu',
        remote_check_root='/check-output',shared_root='/shared',driver_root='/driver')


def test_real_mount_plan_contains_candidate_and_original(tmp_path):
    c=fixture(tmp_path);_,mounts=validate(c)
    assert ('/check-output/package','/candidate') in mounts
    assert ('/original','/original') in mounts
    ast.parse(RUNNER)


@pytest.mark.parametrize('fault',['bytes','manifest','overlap','missing_mount','unsafe_path','symlink'])
def test_bad_inputs_rejected_before_remote(tmp_path,fault):
    c=fixture(tmp_path);p=tmp_path/'package'
    if fault=='bytes':(p/'config.json').write_text('{}')
    if fault=='manifest':c['package_sha256']='wrong'
    if fault=='overlap':c['remote_check_root']='/original/child'
    if fault=='missing_mount':c['shared_root']='/different'
    if fault=='unsafe_path':c['remote_check_root']='/check/../original'
    if fault=='symlink':(p/'link').symlink_to(p/'config.json')
    with pytest.raises(ValueError):validate(c)
