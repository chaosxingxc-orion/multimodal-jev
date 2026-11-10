#!/usr/bin/env python3
"""Count full-registry text tokens with the actual local Omni processor.

No audio/weights/answers loaded. These are text-only lower bounds, not complete
multimodal context acceptance or a finalized task prompt.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path

from multimodal_jev.audio2tool_inventory import registry
from multimodal_jev.baseline import atomic_json, file_hash


def definitions(tools, full):
    lines=[]
    for tool in tools:
        lines.append(f"- {tool['signature']}: {tool['description']}")
        if full:
            lines.append(f"  Defaults: {tool['argument_defaults']}")
            lines.append(f"  Constraints: {tool['argument_constraints']}")
    return '\n'.join(lines)


def run(args):
    from transformers import Qwen3OmniMoeProcessor
    tools=registry(args.registry.read_bytes(),args.registry_hash)
    processor=Qwen3OmniMoeProcessor.from_pretrained(args.model,local_files_only=True)
    results={}
    for name, full in [('official_client_compact',False),('complete_definitions',True)]:
        text=definitions(tools,full)
        system=('You are an AI assistant that converts voice commands into tool calls.\n'
                'Listen to the audio and determine which tool should be called.\n'
                'Output ONLY the tool call in the format: tool_name(param1=value1, param2=value2).\n'
                'If no parameters are mentioned, output: tool_name().\n\nAvailable tools:\n'+text)
        # No audio placeholder: exact count for these text turns is a lower
        # bound only. Full audio feature expansion is deliberately not inferred.
        prompt=processor.apply_chat_template([
            dict(role='system',content=system),
            dict(role='user',content=[dict(type='text',text='Convert this voice command to a tool call.')])
        ],tokenize=False,add_generation_prompt=True)
        tokens=processor.tokenizer.encode(prompt,add_special_tokens=False)
        prompt_path=args.output/(name+'.txt');prompt_path.write_text(prompt)
        results[name]=dict(tools=len(tools),text_tokens=len(tokens),prompt_sha256=file_hash(prompt_path),
            remaining_for_audio_and_extra_text={str(context):context-1024-len(tokens)
                                                for context in (4096,8192,16384,32768)})
    result=dict(registry_sha256=args.registry_hash,model=str(args.model),output_budget_for_arithmetic=1024,
        versions={p:importlib.metadata.version(p) for p in ('transformers','tokenizers')},
        processor_assets={p.name:file_hash(p) for p in args.model.iterdir()
                          if p.is_file() and p.suffix in ('.json','.txt')},results=results,
        runner_sha256=file_hash(__file__),model_loaded=False,audio_read=False,answers_read=False,
        scope='Full candidate registry text-only token lower bounds. No multimodal fit, engine memory, quality, or paper prompt equivalence claim.')
    atomic_json(args.output/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('registry','model','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--registry-hash',required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    try:run(args)
    except BaseException as exc:
        atomic_json(args.output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise
    atomic_json(args.output/'terminal.json',dict(exit_code=0))
