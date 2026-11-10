# Direct audio teacher v3: respond to rejected proposals

V2 completed with43140 tokens and two proposals, both rejected before actor
evaluation: they conflated output labels with typed values and imposed a
fixed `none=unspecified` domain rule. Preserve proposals, review and expense.
Register one new teacher request using the same TRAIN-only evidence/context
plus both proposals and their rejection reasons. No selection/test scores
or predictions enter this request. This is a feedback iteration, not editing
the old response or repeating an unknown call.

Same pinned glm-5.3-flash executor, enabled thinking, stream, max32768 tokens,
900s total,180s socket timeout, no automatic retries. At most2 instructions,
each <=1500 characters, at most3 diagnoses. Only generic matching and label
lookup procedures; no task-specific component rules, synonyms, example answers
or redefining unknown/native none. The actor must output the listed `label`
field even for boolean/null alternatives, never the `value` field. Rendered
task definition and model/audio/header/readout remain identical.

Local structural/semantic checks precede inference. Evaluate every eligible
candidate on the exact immutable parent snapshot1f60533, same1536 requests,
seed17/BF16/eager/frozen, physical175 card4 sequentially. Each3600s app/3900s
outer. Parent completed control retained; no recompute unless identity mismatch.
V1 selection rule and source groups unchanged. Record failures and costs even
if no candidate beats parent. No new weight training or original TF-GRPO claim.
This protocol schedules no further teacher request after terminal result.
