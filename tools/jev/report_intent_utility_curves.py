#!/usr/bin/env python3
"""Report every registered objective update, never a selected best checkpoint.

Read-only: existing controllers and frozen source snapshots are not modified.
Actual per-update Docker/state/backend authentication precedes each diagnostic.
"""
import argparse
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import validate_lineage
from prepare_intent_utility_curves import validate_protocol
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from report_intent_update_metrics import run as update_report


# Objective is the only optimization factor allowed to differ across arms.
VARIABLES = {'pins','objective','step','initial_adapter','acceptance_root','previous_root','previous_suite'}


def run(protocol_path,digest):
    protocol_path=Path(protocol_path)
    if file_hash(protocol_path)!=digest:raise ValueError('frozen curve protocol identity differs')
    protocol=read(protocol_path);pilot_path=Path(protocol['pilot_protocol'])
    if file_hash(pilot_path)!=protocol['pilot_protocol_sha256']:raise ValueError('original pilot changed')
    validate_protocol(protocol,read(pilot_path))
    factors=None;plans=[];image=None;source=None
    # Preflight all 36 pairs before expensive authentication; incomplete is not a result.
    for arm in protocol['arms']:
        prior=None;prior_step=None
        for index,step in enumerate(arm['steps']):
            training,suite=Path(step['training_root']),Path(step['suite_root'])
            for root in (training,suite):
                for name in ('launch.json','config.json','source-manifest.json'):
                    if not (root/name).is_file():raise ValueError('complete registered curves required: '+str(root/name))
                launch=read(root/'launch.json')
                if image is None:image=launch['image']
                if launch['image']!=image:raise ValueError('matched fixed image required')
                check_pins({str(root/'config.json'):launch['config_sha256'],
                            str(root/'source-manifest.json'):launch['source_manifest_sha256']})
                if source is None:source=launch['source_manifest_sha256']
                if launch['source_manifest_sha256']!=source:raise ValueError('matched frozen computation required')
            cfg=read(training/'config.json')
            if (cfg['arm']!='D' or cfg['step']!=index or cfg['objective']!=arm['objective']
                    or any(cfg[k]!=protocol[k] for k in ('pilot','manifest_sha256','seed','lr','global_requests'))):
                raise ValueError('registered objective, population or budget differs')
            current={k:v for k,v in cfg.items() if k not in VARIABLES}
            if factors is None:factors=current
            if factors!=current:raise ValueError('non-objective optimization factor differs')
            if index==0:
                if (cfg['initial_adapter']!=protocol['initial_adapter'] or cfg['acceptance_root']!=protocol['initial_gate']
                        or cfg.get('previous_root') or cfg.get('previous_suite') or cfg.get('initial_optimizer_checkpoint')):
                    raise ValueError('common fresh initialization required')
            else:
                validate_lineage(prior,cfg)
                if cfg['previous_root']!=prior_step['training_root'] or cfg['previous_suite']!=prior_step['suite_root']:
                    raise ValueError('contiguous registered state required')
            selection=read(training/'run/selection.json')
            if selection['recording_ids']!=protocol['schedule'][index]:raise ValueError('registered source schedule differs')
            plans.append((arm['name'],index,training,suite));prior,prior_step=cfg,step
    curves={a['name']:[] for a in protocol['arms']};files={}
    for name,index,training,suite in plans:
        report=update_report(training,suite)
        if report['training_trace']['updates']!=index+1:raise ValueError('accepted cursor differs')
        files.update(report['files'])
        curves[name].append(dict(update=index+1,source_uses=(index+1)*protocol['global_requests'],
            groups=report['groups'],training_trace=report['training_trace']))
    for path in (protocol_path,pilot_path,Path(__file__).resolve()):files[str(path)]=file_hash(path)
    return dict(complete=True,role='learning',updates=12,protocol_sha256=digest,curves=curves,files=files,
        scope='All three objectives and all12 accepted updates. Each point includes before/after margins and correctness, '
              'post-update KL and pre-update gradient/clipping traces. Read alongside full classification curves; '
              'not parameter displacement, independent efficacy or transfer. No best-point selection.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('immutable complete report exists')
    atomic_json(a.output,run(a.protocol,a.protocol_sha256))
