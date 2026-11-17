# Native update_policy microbatch weighting gate

Before execution, register this **loop/loss fixture**, separate from the queued
real-Qwen native update_policy gate. Use all48 saved real responses from
`vllm-pair-signal-n4-v1/run`, full-group pair rewards with weight.2, pinned verl
standardized GRPO, and their actual response lengths and old log probabilities.
No new sampling, correctness selection or model training is performed.

Use a deterministic three-parameter perturbation of saved log probabilities
as the forward fixture. Invoke actual pinned `DataParallelPPOActor.update_policy`
with one epoch, one minibatch, native gradient clipping and AdamW. Replace only
the instance's microbatch forward with the named fixture and explicitly adapt
CPU placement. The reference is vanilla verl loss on the full batch, one
backward, clip1, AdamW LR1e-5, zero decay, foreach/fused disabled. Start every
comparison from identical parameter and optimizer states.

Compare full-batch native vanilla,8-response microbatches, and12-response
microbatches with registered `jev_static_token_mean_v1`. Require preclip gradients,
parameters and both AdamW moments to match within atol2e-6/rtol2e-4. The adapter
uses the true full token denominator and multiplies by accumulation count to
cancel exactly the native loop's division. An unadapted8-response native
vanilla negative control must have gradient error>1e-5. Preserve all outcomes.

The adapter rejects multiple ranks, dynamic batching, nondivisible row counts,
multiple epochs, KL/entropy terms, rollout importance weights, substituted old
log probabilities and existing global normalization. Callers must supply exact
batch token and response counts and compute advantages before splitting.
This gate validates actual native loop scaling, not ASR forward, Ray/FSDP,
NPU training, no-op optimizer behavior or efficacy. Production use additionally
needs the real-Qwen microbatch gradient/state gate and verified batch admission.

## Completed fixture result

The [receipt](../../../../docs/engineering/native-verl-microbatch-loss-v1-result.json)
records exact Docker/application exit0 and independent saved-tensor comparisons.
There are299 response tokens. Native full-batch results match the reference
exactly. With the registered adapter,8-row and12-row microbatches have maximum
preclip gradient errors4.190951585769653e-9 and1.862645149230957e-9 respectively;
parameters and both moments pass the original tolerances. The8-row chunks have
52/51/48/49/51/48 effective tokens, so equal row count does not mean equal tokens.

The unadapted local-mean negative control has gradient error0.0005456390790641308
and first-moment error5.4564094170928e-5. Its first AdamW parameter update is
nevertheless identical in this fixture: checking updated parameters alone would
miss this scaling error. No inference about the ASR model's microbatch numerical
parity follows. Invalid configurations, multiple ranks, invalid masks and loss
registry collisions are covered by16 passing system-torch admission tests.
