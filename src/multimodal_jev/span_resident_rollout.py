"""Resident TP1 Span inference across immutable BF16 LoRA exports.

This owns an engine, not device leases or training admission. The caller supplies
independently computed HF references and authenticated, label-free inputs. Any
evaluation failure poisons the session; it never retries an uncertain switch.
"""
import hashlib
import json
from pathlib import Path
from .policy_identity import checkpoint_identity
from .span_parity import compare_span_policies
from .span_candidate_transport import unpack_candidate_logits
from .span_request_identity import tensor_identity


def adapter_layout(path):
    import torch
    from safetensors import safe_open
    path = Path(path)
    targets = json.loads((path/'adapter_config.json').read_text())['target_modules']
    if (not isinstance(targets, list) or not targets or len(set(targets)) != len(targets)
            or any(not t.startswith('model.language_model.layers.')
                   or t.rsplit('.', 1)[-1] not in ('q_proj', 'v_proj') for t in targets)):
        raise ValueError('explicit distinct language q/v LoRA targets required')
    with safe_open(str(path/'adapter_model.safetensors'), framework='pt', device='cpu') as saved:
        keys = list(saved.keys())
        if not keys or any(saved.get_tensor(k).dtype != torch.bfloat16 for k in keys):
            raise ValueError('nonempty BF16 adapter export required')
    return sorted(targets)


def create_engine(config, targets):
    import torch
    from vllm import LLM
    from .vllm_spans import register_span_model
    from .vllm_audio_features import enable_feature_passthrough
    register_span_model(); enable_feature_passthrough()
    packed = sorted({t.replace('model.language_model.', 'language_model.model.', 1).rsplit('.', 1)[0]+'.qkv_proj' for t in targets})
    return LLM(model=config['model'], tokenizer=config['model'], runner='pooling',
        hf_overrides={'architectures': ['Qwen3ASRForSpanDecisions'], 'head_dtype': 'model'},
        pooler_config={'task': 'token_classify', 'tok_pooling_type': 'ALL', 'use_activation': False},
        dtype='bfloat16', max_model_len=4096, gpu_memory_utilization=.3,
        tensor_parallel_size=1, enforce_eager=True, max_num_seqs=1, seed=17,
        worker_extension_cls='multimodal_jev.lora_span_audit.SpanAuditWorkerExtension',
        enable_chunked_prefill=False, enable_prefix_caching=False, enable_mm_embeds=True,
        limit_mm_per_prompt={'audio': 1}, enable_lora=True, max_loras=1, max_cpu_loras=1,
        max_lora_rank=8, lora_dtype=torch.bfloat16, lora_target_modules=packed)


class ResidentSpanRollout:
    def __init__(self, config, initial_adapter):
        from .span_runtime_profile import install_rollout_profile
        if config['max_model_len'] != 4096 or config['physical_devices'] != [4]:
            raise ValueError('fixed TP1 physical NPU4 and context4096 required')
        if sorted(p.name for p in Path('/dev').glob('davinci[0-9]*')) != ['davinci4']:
            raise ValueError('isolated physical NPU4 required')
        if checkpoint_identity(config['model'])[0] != config['base_id']:
            raise ValueError('base identity differs')
        self.config = {k: config[k] for k in ('model','base_id','runtime_profile','max_model_len')}
        self.targets = adapter_layout(initial_adapter)
        self.slots = {}; self.phase = 'starting'
        self.engine = create_engine(config, self.targets)
        try:
            self.runtime_receipt = install_rollout_profile(self.engine, config['runtime_profile'])
        except BaseException:
            self.phase = 'failed'; self.close(); raise
        self.phase = 'ready'

    def evaluate(self, config, compiled, reference, progress=None):
        """Evaluate one complete policy population; retain the same engine.

        compiled yields the standard prepared Span rows. Targets/rewards are
        never part of this interface. Returned parity is not training admission:
        caller must authenticate reference provenance, process exits and states.
        """
        if self.phase != 'ready':
            raise ValueError('ready resident rollout required; no retry after failure')
        self.phase = 'evaluating'
        try:
            result = self._evaluate(config, compiled, reference, progress)
        except BaseException:
            self.phase = 'failed'; raise
        self.phase = 'ready'
        return result

    def _evaluate(self, config, compiled, reference, progress):
        import torch
        from vllm import PoolingParams
        from vllm.lora.request import LoRARequest
        if any(config[k] != v for k,v in self.config.items()):
            raise ValueError('resident runtime identity differs')
        adapter = config['adapter']; adapter_id = config['adapter_id']
        if checkpoint_identity(adapter)[0] != adapter_id or adapter_layout(adapter) != self.targets:
            raise ValueError('immutable adapter identity or layout differs')
        if (reference.get('passed') is not True or reference['base_id'] != config['base_id']
                or reference['adapter_id'] != adapter_id):
            raise ValueError('reference policy identity differs')
        refs = {r['id']: r for r in reference['records']}
        count = config['population']['requests']
        if type(count) is not int or count < 1 or len(refs) != count or len(reference['records']) != count:
            raise ValueError('complete distinct reference population required')
        if adapter_id not in self.slots:
            self.slots[adapter_id] = len(self.slots)+1
        slot = self.slots[adapter_id]
        lora = LoRARequest('resident-'+adapter_id, slot, str(adapter))
        rows = []; seen = set()
        for row,batch,positions,candidates,rid,identity,processor in compiled:
            uid = row['id']
            if uid in seen or uid not in refs:
                raise ValueError('duplicate or unexpected input')
            ref = refs[uid]
            if ref['request_id'] != rid or ref['request_identity'] != identity:
                raise ValueError('policy-bound request identity differs')
            serialized = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            if (hashlib.sha256(serialized).hexdigest() != rid
                    or identity['policy_id'] != config['base_id']+':'+adapter_id
                    or identity['candidate_ids'] != candidates
                    or identity['positions'] != positions.tolist()
                    or identity['inputs'] != {k:tensor_identity(v) for k,v in sorted(batch.items())}
                    or identity['sampling'] != dict(temperature=1.0,policy='independent_finite_categorical')):
                raise ValueError('compiled tensors or support differ from bound identity')
            seen.add(uid)
            ids = batch['input_ids'][0].tolist()
            if len(ids) > self.config['max_model_len']:
                raise ValueError('full input exceeds context')
            audio_id = processor.tokenizer.convert_tokens_to_ids('<|audio_pad|>')
            at = [i for i,x in enumerate(ids) if x == audio_id]
            if not at or at != list(range(at[0],at[-1]+1)):
                raise ValueError('contiguous audio tokens required')
            compact = ids[:at[0]]+[audio_id]+ids[at[-1]+1:]
            frames = int(batch['input_features_mask'].sum())
            data = {'input_audio_features': batch['input_features'][0,:,:frames].contiguous(),
                    'audio_feature_lengths': torch.tensor([frames]),
                    'feature_attention_mask': batch['input_features_mask'][:,:frames].contiguous()}
            prompt = {'prompt_token_ids': compact, 'multi_modal_data': {'audio': data},
                      'multi_modal_uuids': {'audio': [rid]}}
            params = PoolingParams(use_activation=False, extra_kwargs={'span_positions': positions[0].tolist(),
                'candidate_ids': candidates, 'expected_input_ids': ids})
            result = self.engine.encode([prompt], pooling_params=params, pooling_task='token_classify',
                use_tqdm=False, lora_request=lora)[0]
            if result.prompt_token_ids != ids:
                raise ValueError('expanded input differs')
            logp = [p.tolist() for p in unpack_candidate_logits(result.outputs.data,candidates)]
            parity = compare_span_policies(ref['candidate_logprobs'],logp)
            if not parity['passed']:
                raise ValueError('resident full-candidate parity failed')
            rows.append(dict(id=uid,request_id=rid,request_identity=identity,candidate_logprobs=logp,parity=parity))
            if progress is not None: progress(len(rows),count)
        if seen != refs.keys():
            raise ValueError('incomplete input population')
        worker = self.engine.llm_engine.collective_rpc('inspect_jev_span_lora')
        if (len(worker) != 1 or worker[0]['registered_adapter_ids'] != [slot]
                or worker[0]['lora_dtype'] != 'torch.bfloat16'
                or worker[0]['head_projection_dtype'] != 'torch.bfloat16'
                or not worker[0]['actual_lora_buffer_dtypes']
                or any(v != ['torch.bfloat16'] for v in worker[0]['actual_lora_buffer_dtypes'].values())):
            raise ValueError('actual worker adapter or BF16 execution differs')
        if checkpoint_identity(adapter)[0] != adapter_id:
            raise ValueError('adapter changed during evaluation')
        return dict(passed=True,base_id=config['base_id'],adapter_id=adapter_id,slot=slot,
                    backend='vllm',worker=worker,records=rows)

    def close(self):
        if self.phase == 'closed': return
        self.engine.llm_engine.engine_core.shutdown(timeout=30.)
        self.phase = 'closed'
