"""Unreleased prefix composition; no source rewriting or dispatch entry.

Legacy transitive packagers remain inventoried debt. This module only attaches
an ordinary shared runner and content-proof providers to a private graph.
"""
import importlib
import json
from pathlib import Path
from tools.reports.objective_parallel_admission import MODULES, clone_graph
from tools.reports import prepare_objective_training_candidate as prefix


def expected(c,p,*,original):
    files,m=original(c,p)
    for name,source in [('runtime-startup.py','context_runtime_startup.py'),
                        ('run.py','context_stage2_runtime_runner.py')]:
        files[name]=Path(__file__).with_name(source).read_bytes()
        if name not in m['layout']['configs']:m['layout']['configs'].append(name)
    config=json.loads(files['config.json']);remote=c['remote_root']
    for name in ('run.py','runtime-startup.py'):
        config['pins'][remote+'/'+name]=prefix.sha(files[name])
    files['config.json']=prefix.encode(config)
    for name in m['layout']['services'].values():
        service=json.loads(files[name]);service['pins']=config['pins'];files[name]=prefix.encode(service)
    m['files']={name:prefix.sha(raw) for name,raw in files.items()}
    return files,m


def graph(engine):
    modules=[importlib.import_module('tools.reports.'+name) for name in MODULES]
    views=clone_graph(modules,{
        ('tools.reports.context_infonce_selection','reconstruct_plan'):engine.reconstruct,
        ('tools.reports.prepare_context_reference_job','replay_reference_outputs'):engine.replay})
    package=views['tools.reports.prepare_objective_training_candidate']
    original=package.expected
    def build(config,provenance):return expected(config,provenance,original=original)
    package.prepare.__globals__['expected']=build;package.expected=build
    return views


def fast_launch(*args,**kwargs):
    raise RuntimeError('retired source-rewriting launcher; composed lifecycle admission required')


def acceptance_graph(engine):
    raise RuntimeError('all-stage composed package and lifecycle admission remains incomplete')
