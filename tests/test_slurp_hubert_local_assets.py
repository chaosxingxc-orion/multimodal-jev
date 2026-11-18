import hashlib
from tools.reports.audit_slurp_hubert_local_assets import scan,content_digest


def test_finds_renamed_blob_and_rejects_same_size_wrong_content(tmp_path):
    (tmp_path/'opaque').write_bytes(b'weights');(tmp_path/'wrong').write_bytes(b'badfile')
    e=[dict(bytes=7,algorithm='sha256',digest=hashlib.sha256(b'weights').hexdigest())]
    r=scan(e,[tmp_path]);assert r['all_core_assets_found'];assert r['assets'][0]['matches']==[str(tmp_path/'opaque')]


def test_git_blob_hash_and_lfs_pointer_not_accepted(tmp_path):
    p=tmp_path/'tokenizer';p.write_bytes(b'abc')
    assert content_digest(p,'git_blob_sha1')==hashlib.sha1(b'blob 3\0abc').hexdigest()
    (tmp_path/'model').write_text('version https://git-lfs.github.com/spec/v1\n')
    e=[dict(bytes=377575130,algorithm='sha256',digest='0'*64)]
    r=scan(e,[tmp_path]);assert not r['all_core_assets_found'];assert r['missing_bytes']==377575130


def test_missing_root_and_linked_directory_are_explicit(tmp_path):
    target=tmp_path/'target';target.mkdir();(target/'blob').write_bytes(b'x')
    root=tmp_path/'root';root.mkdir();(root/'link').symlink_to(target,target_is_directory=True)
    e=[dict(bytes=1,algorithm='sha256',digest=hashlib.sha256(b'x').hexdigest())]
    r=scan(e,[root,tmp_path/'absent']);assert not r['all_core_assets_found'];assert r['missing_roots']
