"""Reviewed learning-only audio Span SFT; no autoregressive decoding."""
from collections import defaultdict
import json
from pathlib import Path
import random
import time

from .baseline import atomic_json, file_hash
from .slurp_training import validate_admission
from .span_contract import SpanQuestion, SpanRequest, encode_span_request, span_labels
from .task_schema import DecisionSchema, JevExample


def load_reviewed_bundles(release, expected_manifest_sha256, question_order):
    """Group only authenticated admitted labels; absent labels stay masked."""
    release = Path(release)
    if file_hash(release / 'MANIFEST.json') != expected_manifest_sha256:
        raise ValueError('reviewed release manifest changed')
    manifest = json.loads((release / 'MANIFEST.json').read_text())
    if manifest['schema'] != 'slurp-reviewed-training-release-v1':
        raise ValueError('reviewed learning release required')
    for name, digest in manifest['files'].items():
        if file_hash(release / name) != digest:
            raise ValueError('reviewed release file changed: ' + name)
    tasks = json.loads((release / 'tasks.json').read_text())
    if not question_order or len(set(question_order)) != len(question_order) or set(question_order) != set(tasks):
        raise ValueError('complete unique fixed question order required')
    questions = tuple(SpanQuestion(k, DecisionSchema.from_dict(tasks[k]['schema'])) for k in question_order)
    requirements = tuple(tasks[question_order[0]]['thinking']['requirements'])
    if any(tuple(t['thinking']['requirements']) != requirements for t in tasks.values()):
        raise ValueError('shared thinking requirements required')
    grouped = defaultdict(dict)
    for line in (release / 'train.jsonl').read_text().splitlines():
        ex = JevExample.from_record(json.loads(line))
        validate_admission(ex)
        recording, task = ex.example_id.rsplit('::', 1)
        if task not in tasks or task in grouped[recording]:
            raise ValueError('unknown or duplicated decision')
        if (ex.schema != DecisionSchema.from_dict(tasks[task]['schema']) or
                tuple(ex.thinking.requirements) != requirements):
            raise ValueError('admitted task differs from fixed question bank')
        grouped[recording][task] = ex
    bundles = []
    for recording, decisions in sorted(grouped.items()):
        first = next(iter(decisions.values()))
        identity = (first.audio, first.metadata['audio_sha256'], first.metadata['source_group'])
        if any((e.audio, e.metadata['audio_sha256'], e.metadata['source_group']) != identity for e in decisions.values()):
            raise ValueError('mixed recording or source identity')
        if file_hash(first.audio) != identity[1]:
            raise ValueError('audio changed')
        request = SpanRequest(first.audio, questions, question_order[-1], requirements)
        # Targets and provenance remain outside the model input.
        targets = [decisions[k].target if k in decisions else None for k in question_order]
        bundles.append(dict(id=recording, request=request, targets=targets,
                            supervised=[k in decisions for k in question_order],
                            audio_sha256=identity[1], source_group=identity[2],
                            decision_ids=[decisions[k].example_id for k in question_order if k in decisions]))
    if (len(bundles) != manifest['reviewed_recordings'] or
            sum(sum(b['supervised']) for b in bundles) != manifest['decisions']):
        raise ValueError('reviewed population count mismatch')
    return bundles, manifest


def target_tokens(bundle, mappings):
    targets = []
    for value, active, mapping in zip(bundle['targets'], bundle['supervised'], mappings, strict=True):
        if not active:
            targets.append(-100)
            continue
        matches = [i for i, candidate in enumerate(mapping['values'])
                   if type(candidate) is type(value) and candidate == value]
        if len(matches) != 1:
            raise ValueError('supervised value absent from legal candidates')
        targets.append(mapping['token_ids'][matches[0]])
    return targets


def train(config, output):
    import torch
    from .asr import load_runtime
    from .audio_policy import load_audio
    from .sft import initialize_adapter
    from .span_model import NativeAudioSpanModel, span_supervised_loss, span_candidate_loss
    output = Path(output)
    objective = config.get('objective','full_vocabulary_ce')
    if objective not in {'full_vocabulary_ce','candidate_ce'}:
        raise ValueError('unsupported Span training objective')
    bundles, manifest = load_reviewed_bundles(config['release'], config['manifest_sha256'], config['question_order'])
    if config.get('sampling_mode') == 'joint_uniform_complement_pairs_v1':
        if config['recordings_per_update'] != 2:
            raise ValueError('complement pairs require two recordings per update')
        from .span_balance import balanced_binary_batches
        schedule, balance = balanced_binary_batches(bundles,config['seed'])
        atomic_json(output / 'balance.json',balance)
    elif config.get('sampling_mode', 'shuffle') == 'shuffle':
        order = list(range(len(bundles)))
        random.Random(config['seed']).shuffle(order)
        size = config['recordings_per_update']
        schedule = [order[i:i+size] for i in range(0,len(order),size)]
    else:
        raise ValueError('unsupported sampling mode')
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    torch.npu.manual_seed_all(config['seed'])
    processor, base = load_runtime(config['model_path'], 'npu:0', 'bfloat16')
    adapted = initialize_adapter(base, config).eval()
    model = NativeAudioSpanModel(adapted.get_base_model()).eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError('no trainable adapter parameters')
    prepared = []
    audio_receipts = []
    for bundle in bundles:
        audio, receipt = load_audio(bundle['request'].audio, config.get('audio_preprocessing'), return_receipt=True)
        audio_receipts.append(receipt)
        inputs, positions = encode_span_request(processor, bundle['request'], audio)
        mappings = span_labels(processor.tokenizer, bundle['request'].questions)
        prepared.append((inputs, positions, mappings, target_tokens(bundle, mappings)))
    atomic_json(output / 'identity.json', dict(config=config, review_scope=manifest['review_scope'],
        model_weights_sha256=file_hash(Path(config['model_path']) / 'model.safetensors'),
        source_sha256={name:file_hash(Path(__file__).with_name(name)) for name in
            ('span_training.py','span_contract.py','span_model.py','span_balance.py','sft.py','asr.py','audio_policy.py','audio_preprocess.py')},
        bundles=[dict(id=b['id'], source_group=b['source_group'], audio_sha256=b['audio_sha256'],
                      decision_ids=b['decision_ids'], supervised=b['supervised'],
                      targets=b['targets'], request=b['request'].model_input(), audio_processing=audio_receipts[i],
                      positions=prepared[i][1].tolist(), input_ids=prepared[i][0]['input_ids'].tolist())
                 for i,b in enumerate(bundles)]))

    def forward(index, silence=False):
        inputs, positions, mappings, targets = prepared[index]
        if silence:
            import numpy as np
            inputs, positions = encode_span_request(processor, bundles[index]['request'],
                np.zeros(audio_receipts[index]['output_samples'],dtype=np.float32))
        inputs = {k:v.to(device='npu:0', dtype=model.lm_head.weight.dtype if v.is_floating_point() else v.dtype)
                  for k,v in inputs.items()}
        vocab, logp = model(inputs, positions.to('npu:0'), [m['token_ids'] for m in mappings])
        labels = torch.tensor([targets], device='npu:0')
        mask = torch.tensor([bundles[index]['supervised']], device='npu:0', dtype=torch.bool)
        return vocab, logp, labels, mask

    @torch.inference_mode()
    def evaluate(name, silence=False):
        records = []
        for i,b in enumerate(bundles):
            _, logp, _, _ = forward(i, silence)
            mappings = prepared[i][2]
            predictions = [m['values'][int(p.argmax(-1).item())] for m,p in zip(mappings,logp)]
            correct = [pred == gold if active else None for pred,gold,active in
                       zip(predictions,b['targets'],b['supervised'])]
            records.append(dict(id=b['id'], predictions=predictions, correct=correct,
                                candidate_logprobs=[p.cpu().tolist() for p in logp]))
            atomic_json(output / 'progress.json', dict(phase=name, completed=i+1,total=len(bundles), heartbeat_unix=time.time()))
        per_span = {}
        for j,task_name in enumerate(config['question_order']):
            confusion = [[0,0],[0,0]]
            for b,r in zip(bundles,records):
                if b['supervised'][j]:
                    confusion[int(b['targets'][j])][int(r['predictions'][j])] += 1
            recalls = [confusion[k][k]/sum(confusion[k]) if sum(confusion[k]) else None for k in (0,1)]
            per_span[task_name] = dict(confusion_gold_by_prediction=confusion,recall_false=recalls[0],
                                 recall_true=recalls[1],balanced_accuracy=sum(recalls)/2 if None not in recalls else None)
        result = dict(records=records, per_span=per_span, decisions=sum(sum(b['supervised']) for b in bundles),
                      correct=sum(c is True for r in records for c in r['correct']),
                      complete_recordings=sum(all(b['supervised']) for b in bundles),
                      complete_correct=sum(all(c is True for c in r['correct']) for r in records),
                      scope='Learning-population diagnostic only; no held-out accuracy or CoT benefit claim.')
        atomic_json(output / (name + '.json'), result)
        return {k:v for k,v in result.items() if k != 'records'}

    before = evaluate('before')
    optimizer = torch.optim.AdamW(parameters, lr=config['lr'], weight_decay=0., foreach=False, fused=False)
    history = []
    for batch in schedule:
        denominator = sum(sum(bundles[i]['supervised']) for i in batch)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for i in batch:
            vocab, logp, labels, mask = forward(i)
            loss = (span_supervised_loss(vocab, labels, mask) if objective == 'full_vocabulary_ce' else
                    span_candidate_loss(logp, labels, mask, [m['token_ids'] for m in prepared[i][2]]))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite span loss')
            (loss * (int(mask.sum()) / denominator)).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if float(norm) == 0:
            raise ValueError('no adapter gradient')
        optimizer.step()
        history.append(dict(step=len(history)+1, recordings=[bundles[i]['id'] for i in batch],
                            supervised_spans=denominator, per_recording_losses=losses, gradient_norm=float(norm)))
        atomic_json(output / 'progress.json', dict(phase='train', updates=len(history), heartbeat_unix=time.time()))
        atomic_json(output / 'history.json', history)
    adapted.save_pretrained(output / 'adapter')
    processor.save_pretrained(output / 'processor')
    for folder in ('adapter', 'processor'):
        for path in (output / folder).iterdir():
            if path.is_file():
                path.chmod(path.stat().st_mode | 0o444)
    after = evaluate('after')
    silence = evaluate('after-silence', silence=True)
    return dict(status='passed', updates=len(history), recordings=len(bundles),
                training_recordings=len({i for batch in schedule for i in batch}),
                supervised_spans=sum(sum(bundles[i]['supervised']) for batch in schedule for i in batch),
                diagnostic_spans=manifest['decisions'], head='pretrained_native_lm_head',
                objective=objective, forward_contract='one audio backbone forward per recording',
                before=before, after=after, after_silence=silence,
                adapter_files={p.name:file_hash(p) for p in (output/'adapter').iterdir() if p.is_file()},
                scope='One epoch multi-span SFT wiring and learning diagnostics. Parallel roles; not CoT, RL, held-out gains or academic reproduction.')
