"""Reference client overlay; the pinned numerical runtime stays intact."""
from reference_resume_builder import main
from reference_resume_pool import ResumePool, lineage


def resume(pool, build, endpoint_sha256):
    return ResumePool(pool, **build['resume'], lineage=lineage(build, endpoint_sha256=endpoint_sha256))


if __name__ == '__main__':
    main(resume_pool=resume)
