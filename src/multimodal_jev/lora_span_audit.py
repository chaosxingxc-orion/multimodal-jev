"""Read-only worker receipt for the explicitly restricted native Span LoRA."""


class SpanAuditWorkerExtension:
    """Named worker RPC: no callable/pickle serialization across processes."""

    def inspect_jev_span_lora(self):
        return inspect_span_lora(self.get_model())

    def align_jev_language_norms(self):
        from .span_norm_alignment import install_hf_order_language_norm
        return install_hf_order_language_norm(self.get_model())

    def align_jev_conv_layout(self):
        from .span_conv_alignment import install_hf_conv_layout
        return install_hf_conv_layout(self.get_model())

    def align_jev_language_attention(self):
        from .span_attention_alignment import install_eager_span_attention
        return install_eager_span_attention(self.get_model())

    def align_jev_language_rotary(self):
        from .span_rotary_alignment import install_hf_order_rotary
        return install_hf_order_rotary(self.get_model())

    def align_jev_language_qkv(self):
        from .span_projection_alignment import install_separate_qkv
        return install_separate_qkv(self.get_model())

    def align_jev_language_mlp(self):
        from .span_projection_alignment import install_separate_mlp
        return install_separate_mlp(self.get_model())

    def begin_jev_span_trace(self):
        from .span_trace import attach_vllm_trace
        if getattr(self,'_jev_span_trace',None) is not None:
            raise ValueError('trace already active')
        self._jev_span_trace = attach_vllm_trace(self.get_model())
        return {'trace':'attached'}

    def finish_jev_span_trace(self, path):
        import torch
        trace = self._jev_span_trace
        trace.close()
        # Path is supplied only by the trusted local diagnostic launcher.
        torch.save(trace.tensors,path)
        self._jev_span_trace = None
        return {'trace_points':list(trace.tensors),'path':path}


def inspect_span_lora(model):
    from vllm.lora.layers import BaseLayerWithLoRA
    manager=getattr(model,'lora_manager',None)
    if manager is None:
        raise ValueError('worker has no LoRA manager')
    wrapped={name:type(module).__name__ for name,module in model.named_modules()
             if isinstance(module,BaseLayerWithLoRA)}
    expected=set(manager.lora_config.target_modules or [])
    if not expected or set(wrapped)!=expected:
        raise ValueError('worker wrapper coverage differs from explicit targets')
    if any(not name.startswith('language_model.model.layers.') or not name.endswith('.qkv_proj') for name in wrapped):
        raise ValueError('unexpected LoRA wrapper outside language qkv')
    modules = dict(model.named_modules())
    storage = {}
    for name in wrapped:
        tensors = [tensor for attr in ('lora_a_stacked','lora_b_stacked')
                   for tensor in getattr(modules[name],attr,())]
        if not tensors or any(t.dtype != manager.lora_config.lora_dtype for t in tensors):
            raise ValueError('actual worker LoRA buffers differ from configured dtype')
        storage[name] = sorted({str(t.dtype) for t in tensors})
    return dict(model_class=type(model).__name__,wrapper_modules=wrapped,
                actual_lora_buffer_dtypes=storage,
                lora_dtype=str(manager.lora_config.lora_dtype),max_rank=manager.lora_config.max_lora_rank,
                lm_head_class=type(model.language_model.lm_head).__name__,
                logits_processor_class=type(model.language_model.logits_processor).__name__,
                head_projection_dtype=str(model.language_model.logits_processor.head_dtype),
                registered_adapter_ids=sorted(manager.list_adapters()),
                scope='Worker module coverage/configuration; file pin and output effect checked separately.')
