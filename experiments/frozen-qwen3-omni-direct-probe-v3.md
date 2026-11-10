# Qwen3-Omni v3: explicitly release engine before declaring application success

V2 spawn solved worker initialization. All15weight shards loaded and all6
requests generated;3strict JSON,3Markdown-fenced invalid,none truncated.
Weight rehash passed and application terminal0 was written, but the process
and engine workers stayed alive during interpreter teardown. Thus v2 has no
successful container terminal and cannot pass the registered gate.

Inspect exact live container, preserve prediction/result/app terminal hashes,
record manual stop of this completed-but-not-exited attempt; never rewrite its
terminal to a pass. V3 changes only lifecycle: call the pinned engine client's
shutdown(timeout=30) after generating6requests, before final model verification
and success marker. This method is present in the fixed vLLM core_client.py.
All model/audio/prompt/seed/BF16/TP2/eager/context/generation settings remain
identical; retain spawn, same image and physical4/5. New immutable snapshot,
new root,one attempt3600s. Both app/container success still required. Do not
strip Markdown, alter prompt or claim task efficacy from this one group.
