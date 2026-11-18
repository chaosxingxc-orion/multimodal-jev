"""Separate verl entrypoint with configurable within-update bucket allocation."""
import argparse
import ast
import json
from pathlib import Path
from tools.reports.context_macro_trainer_candidate import generate as parent_generate, repartition as original
from multimodal_jev.context_length_buckets import plan


def repartition(lengths,frozen_ranks,*,microbatch,world_size,settings):
    original(lengths,frozen_ranks,microbatch=microbatch,world_size=world_size)
    return plan(lengths,microbatch=microbatch,world_size=world_size,**settings)[0]


def generate(root):
    source=parent_generate(root)
    before='from tools.reports.context_macro_trainer_candidate import topology,repartition'
    after='from tools.reports.context_macro_trainer_candidate import topology\n    from tools.reports.context_bucket_trainer_candidate import repartition'
    call="planned=repartition(lengths,batches[index]['rank_indices'],microbatch=c['microbatch'],world_size=world)"
    if source.count(before)!=1 or source.count(call)!=1:
        raise ValueError('fixed parent integration anchor changed')
    source=source.replace(before,after).replace(call,call[:-1]+",settings=c['length_buckets'])")
    # Bind schedule strategy to checkpoints as well as configuration manifests.
    anchor="macro_layout=layout,schedule_release_sha256="
    if source.count(anchor)!=1:raise ValueError('checkpoint layout anchor changed')
    source=source.replace(anchor,"length_buckets=c['length_buckets'],macro_layout=layout,schedule_release_sha256=")
    ast.parse(source);return source


def run(config,output):
    if set(config['length_buckets'])!={'token_bucket_width','frame_bucket_width','token_cost','frame_cost'}:
        raise ValueError('explicit fixed bucket settings required')
    ns={'__name__':__name__+'.derived'}
    exec(compile(generate(config['source_root']),'<bucket-verl-trainer>','exec'),ns)
    return ns['run'](config,Path(output))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.output)
