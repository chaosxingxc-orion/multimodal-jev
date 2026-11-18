"""Candidate worker extension: quiescent joint head/native-LoRA activation."""
import torch
from multimodal_jev.context_worker_extension import ContextBootstrapWorkerExtension
from multimodal_jev.context_policy_artifacts import read_execution_bundle
from context_policy_activation import HeadActivation,prepare_switch


def verify_loaded_lora(model,bundle,slot):
    manager=model.lora_manager
    if set(manager.list_adapters())!={slot} or manager.lora_index_to_id!=[slot]:
        raise ValueError('one active native adapter slot required')
    loaded=manager.get_adapter(slot)
    prefixes={name.rsplit('.',1)[0] for name in bundle['contract']['lora']}
    expected={p.replace('language_model.','language_model.model.',1)+'.qkv_proj' for p in prefixes}
    if set(loaded.loras)!=expected:raise ValueError('native packed adapter coverage differs')
    count=0
    for prefix in prefixes:
        name=prefix.replace('language_model.','language_model.model.',1)+'.qkv_proj'
        packed=loaded.get_lora(name)
        if not packed.is_packed or len(packed.lora_a)!=3 or len(packed.lora_b)!=3:
            raise ValueError('native q/k/v packing differs')
        for index,suffix in enumerate(('q_proj','k_proj','v_proj')):
            if suffix=='k_proj':
                if packed.lora_a[index] is not None or packed.lora_b[index] is not None:
                    raise ValueError('unexpected key projection adapter')
                continue
            source=prefix+'.'+suffix;spec=bundle['contract']['lora'][source]
            a=bundle['tensors']['backbone.'+source+'.lora_A.default.weight']
            b=bundle['tensors']['backbone.'+source+'.lora_B.default.weight']*spec['scaling']
            if (packed.scaling[index]!=1 or packed.rank!=spec['rank'] or
                    not torch.equal(packed.lora_a[index].detach().cpu(),a) or
                    not torch.equal(packed.lora_b[index].detach().cpu(),b)):
                raise ValueError('native loaded BF16 LoRA tensors/scaling differ')
            count+=2
    return dict(policy_id=bundle['policy_id'],slot=slot,adapter_verified=True),count


class SwitchingContextWorker(ContextBootstrapWorkerExtension):
    def finish_jev_context_startup(self,expected_policy_id):
        receipt=super().finish_jev_context_startup(expected_policy_id)
        model=self.get_model()
        spec=self.vllm_config.model_config.hf_config.jev_context_policy
        bundle=read_execution_bundle(spec['snapshot'],expected_policy_id=expected_policy_id,
            expected_contract=spec['contract'],expected_base_id=spec['base_id'])
        self._jev_activation=HeadActivation(model.pooler.readout,model.language_model.model.embed_tokens.weight,bundle)
        return receipt

    def switch_jev_context_policy(self,expected_policy_id,snapshot,policy_id,adapter,slot):
        tx=self._jev_activation;model=self.get_model()
        try:
            if model._jev_policy_id!=expected_policy_id or model.pooler.memory_profiling:
                raise ValueError('quiescent serving policy required')
            bundle=prepare_switch(tx.current,snapshot=snapshot,policy_id=policy_id,
                adapter=adapter,base_id=tx.current['contract']['base_id'])
            tx.stage(bundle,expected_policy_id=expected_policy_id,slot=slot)
            from vllm.lora.request import LoRARequest
            if not self.add_lora(LoRARequest('context-'+policy_id,slot,adapter)):
                raise ValueError('fresh native adapter ID required')
            receipt,count=verify_loaded_lora(model,bundle,slot)
            result=tx.commit(receipt);model._jev_policy_id=policy_id
            return dict(result,native_loaded_tensors=count,
                        scope='Loaded native adapter CPU tensors and active slot checked; audio probability parity separate')
        except BaseException:
            tx.phase='failed';model.pooler.readout.policy_id='activation-failed';model._jev_policy_id='activation-failed'
            raise
