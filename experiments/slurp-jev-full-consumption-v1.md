# SLURP-Jev full consumption v1

Build from verified native real-audio release240070cb69d71f5df53874ff9e728611dd1bd56a564f7f9e7718092125612add.
All50627/8690/13078 recordings and native linked utterance source groups remain
in their official splits. No semantic filtering, oracle transcription input,
training, teacher call or model inference is dispatched by this data protocol.

One audio bundle per recording with a shared, hashed task definition bank.
The runtime resolves definitions into ordinary Jev requests, never reading gold.
This avoids materializing millions of identical definitions/audio references.
Targets/provenance reside in a separate file and bind to the exact bundle hash.

Per audio, expose168 decisions: joint scenario/action Choice with all60 legal
TRAIN pairs, order permutation and identity rebinding (3); presence and its
negative proposition for every55 TRAIN entity types (110); every occurrence of
each entity type as ordered open_slots string list (55). No gold-dependent
question inclusion or negative sampling; absent and present types are queried
equally. Permutation is derived from source_group and candidate ID, not gold.
Rebind rotates IDs while keeping descriptions; none of these changes labels.

open_slots is an explicitly new generative type, distinct from legacy scalar
open_slot and nga's finite Choice/Noul/Score. Empty list means no occurrence;
null is disallowed here, not a synonym for absence. Keep duplicate same-type
fillers and their ordering. Lowercase fillers exactly as native official spans;
all original spans and raw annotations remain available via native target ID.
Grouped extraction preserves all(type,filler) occurrences but does not retain
ordering between different types; official entity matching is multiset-based.
No invented ordinal Score and no claim that absence is acoustic uncertainty.

Runtime input: audio plus resolved question/schema only. Bundle/source IDs,
filenames, provenance, native labels and references never enter rendered prompts.
Target file maps task IDs to typed values; a source hash pins native provenance.
Task bank contains only TRAIN taxonomy, not per-recording entity or intent gold.
60-way Choice is unsupported by current36-token finite readout; fail explicitly
instead of truncating or substituting oracle scenario-conditioned choices.

Evaluation must require complete requested recording/task coverage and score
intent, three-view both/all-correct, complementary presence both-correct,
ordered slot-list exact, nonempty slot-list exact and reconstructed native
entity metrics separately. Constant true/false, empty-list and first-occurrence
controls must fail appropriate paired/nonempty/multiplicity tests. Micro average
over all168 decisions is not native SLU performance and will be dominated by
absent entities. Only actual model predictions can establish task efficacy.

Existing domain-held-out/first-slot release remains a distinct historical
projection; do not overwrite it or call it complete native SLURP.
