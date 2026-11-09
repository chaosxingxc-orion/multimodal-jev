# 来源与核验方法

调研截至 2026-11-09。优先官方模型卡、配置、代码、原论文全文与原数据发布；“最新”只覆盖本次可核验的公开权重。未把网页搜索片段当全文，也未把代理返回的 HTTP 200 提醒页当有效资料。

未进行模型训练/推理复现；下载状态另见 [本地资源](06-local-assets.md)。论文成绩是作者报告；本文的 SLURP 计数来自官方清单解析。不同日期的 main、论文 v1/v2 和发布卡可能不一致。

原始响应缓存位于未入 Git 的 `.local/research/sources/`；[sources.json](sources.json) 保留 URL、实际 UTC 访问时刻、响应状态、内容 SHA256 和阅读层级。`retrieved` 表示取到响应，不等于独立验证作者所有结论。被阻断的 TypeSafe 文档已单独标记。

## 文中简写

| 编号 | 来源 |
| --- | --- |
| M1 | [qwen-asr-full](https://arxiv.org/html/2601.21337v1) |
| M2 | [qwen-asr-06-config](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json) |
| M3 | [qwen-asr-17-config](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json) |
| M4 | [qwen-asr-repo](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/README.md) |
| M5 | [qwen-asr-hf-card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/raw/main/README.md) |
| M10 | [qwen-audio-jev-full](https://arxiv.org/html/2610.21614v2) |
| R1 | [audiojev-full](https://arxiv.org/html/2610.01293v1) |
| R2 | [qwen-audio-jev-full](https://arxiv.org/html/2610.21614v2) |
| R7 | [qwen-audio-agent](https://raw.githubusercontent.com/QwenAudio/qwen-audio-agent/main/README.md) |
| R8 | [jarvis-full](https://arxiv.org/html/2610.23804v1) |
| R10 | [slam-slu-full](https://arxiv.org/html/2610.20844v1) |

## 全部一手来源索引

| ID | 来源 | 阅读/访问状态 |
| --- | --- | --- |
| acoustic-gating-full | [原始来源](https://arxiv.org/html/2609.32536v1) | full_text / retrieved |
| acoustic-gating | [原始来源](https://arxiv.org/abs/2609.32536) | abstract / retrieved |
| apex | [原始来源](https://arxiv.org/abs/2611.09491) | abstract / retrieved |
| apriel-paper | [原始来源](https://arxiv.org/abs/2610.40738) | abstract / retrieved |
| audio-decision-recent | [原始来源](https://export.arxiv.org/api/query?search_query=all%3A%22audio%22+AND+%28all%3A%22typed+decision%22+OR+all%3A%22decision+model%22%29&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending) | source_or_metadata / retrieved |
| audio-jev-search | [原始来源](https://api.github.com/search/repositories?q=jev+audio&per_page=15) | source_or_metadata / retrieved |
| audio2tool-blobs | [原始来源](https://huggingface.co/api/datasets/RVtech/Audio2Tool?blobs=true) | source_or_metadata / retrieved |
| audio2tool-card | [原始来源](https://huggingface.co/datasets/RVtech/Audio2Tool/raw/main/README.md) | source_or_metadata / retrieved |
| audio2tool-full | [原始来源](https://arxiv.org/html/2604.22821v1) | full_text / retrieved |
| audio2tool-meta | [原始来源](https://huggingface.co/api/datasets/RVtech/Audio2Tool) | source_or_metadata / retrieved |
| audio2tool-paper | [原始来源](https://arxiv.org/abs/2604.22821) | abstract / retrieved |
| audio2tool-site | [原始来源](https://audio2tool.github.io/) | source_or_metadata / retrieved |
| audiojev-arxiv-search | [原始来源](https://export.arxiv.org/api/query?search_query=all%3AAudioJev%20OR%20all%3A%22audio%20decision%22&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending) | source_or_metadata / retrieved |
| audiojev-card | [原始来源](https://huggingface.co/shlv/AudioJev/raw/main/README.md) | source_or_metadata / retrieved |
| audiojev-code | [原始来源](https://raw.githubusercontent.com/SihanLv/AudioJev-Inference/main/src/audiojev_inference/engine.py) | not_read / request_failed |
| audiojev-config | [原始来源](https://huggingface.co/shlv/AudioJev/raw/main/config.json) | source_or_metadata / retrieved |
| audiojev-full | [原始来源](https://arxiv.org/html/2610.01293v1) | full_text / retrieved |
| audiojev-inference-head | [原始来源](https://api.github.com/repos/SihanLv/AudioJev-Inference/commits/main) | source_or_metadata / retrieved |
| audiojev-model-search | [原始来源](https://huggingface.co/api/models?search=AudioJev&limit=30&full=true) | source_or_metadata / retrieved |
| audiojev-paper | [原始来源](https://arxiv.org/abs/2610.01293) | abstract / retrieved |
| audiojev-tree | [原始来源](https://api.github.com/repos/SihanLv/AudioJev-Inference/git/trees/main?recursive=1) | source_or_metadata / retrieved |
| audiojev | [原始来源](https://raw.githubusercontent.com/SihanLv/AudioJev-Inference/main/README.md) | source_or_metadata / retrieved |
| calibration | [原始来源](https://arxiv.org/abs/1706.04599) | abstract / retrieved |
| canary | [原始来源](https://huggingface.co/nvidia/canary-qwen-2.5b/raw/main/README.md) | source_or_metadata / retrieved |
| clozeger-full | [原始来源](https://arxiv.org/html/2405.10025v1) | full_text / retrieved |
| clozeger | [原始来源](https://arxiv.org/abs/2405.10025) | abstract / retrieved |
| community-audiojev-rl | [原始来源](https://huggingface.co/sammyliu/audio-jev-qwen-rlcd-v1/raw/main/README.md) | source_or_metadata / retrieved |
| community-audiojev | [原始来源](https://huggingface.co/sammyliu/audio-jev-qwen-v1/raw/main/README.md) | source_or_metadata / retrieved |
| contextasr | [原始来源](https://arxiv.org/abs/2507.05727) | abstract / retrieved |
| discrete-continuous | [原始来源](https://arxiv.org/abs/2508.17863) | abstract / retrieved |
| download-meta-eva | [原始来源](https://huggingface.co/api/datasets/ServiceNow-AI/eva?blobs=true) | source_or_metadata / retrieved |
| download-meta-macslu | [原始来源](https://huggingface.co/api/datasets/Gatsby1984/MAC_SLU?blobs=true) | source_or_metadata / retrieved |
| download-meta-minds14 | [原始来源](https://huggingface.co/api/datasets/PolyAI/minds14?blobs=true) | source_or_metadata / retrieved |
| download-meta-mmar | [原始来源](https://huggingface.co/api/datasets/BoJack/MMAR?blobs=true) | source_or_metadata / retrieved |
| download-meta-mmau | [原始来源](https://huggingface.co/api/datasets/gamma-lab-umd/MMAU-test?blobs=true) | source_or_metadata / retrieved |
| download-meta-mmsu | [原始来源](https://huggingface.co/api/datasets/ddwang2000/MMSU?blobs=true) | source_or_metadata / retrieved |
| download-meta-slue1 | [原始来源](https://huggingface.co/api/datasets/asapp/slue?blobs=true) | source_or_metadata / retrieved |
| download-meta-slue2 | [原始来源](https://huggingface.co/api/datasets/asapp/slue-phase-2?blobs=true) | source_or_metadata / retrieved |
| download-meta-voicebench | [原始来源](https://huggingface.co/api/datasets/hlt-lab/voicebench?blobs=true) | source_or_metadata / retrieved |
| early-tool-full | [原始来源](https://arxiv.org/html/2610.34653v1) | full_text / retrieved |
| early-tool | [原始来源](https://arxiv.org/abs/2610.34653) | abstract / retrieved |
| espnet-slu | [原始来源](https://arxiv.org/abs/2111.14706) | abstract / retrieved |
| eva-bench | [原始来源](https://arxiv.org/abs/2605.13841) | abstract / retrieved |
| eva-data | [原始来源](https://huggingface.co/datasets/ServiceNow-AI/eva/raw/main/README.md) | source_or_metadata / retrieved |
| eva-full | [原始来源](https://arxiv.org/html/2605.13841v1) | full_text / retrieved |
| eva-head | [原始来源](https://api.github.com/repos/ServiceNow/eva/commits/main) | source_or_metadata / retrieved |
| eva-repo | [原始来源](https://raw.githubusercontent.com/ServiceNow/eva/main/README.md) | source_or_metadata / retrieved |
| fdb-data-download | [原始来源](https://drive.google.com/uc?export=download&id=1SO_4MTazWQ_jvCx0dtmpQ-t40bdd07yz) | not_read / request_failed |
| fdb-full | [原始来源](https://arxiv.org/html/2604.04847v1) | full_text / retrieved |
| fdb-head | [原始来源](https://api.github.com/repos/DanielLin94144/Full-Duplex-Bench/commits/main) | source_or_metadata / retrieved |
| fdb-repo | [原始来源](https://raw.githubusercontent.com/DanielLin94144/Full-Duplex-Bench/main/README.md) | source_or_metadata / retrieved |
| fdb-site | [原始来源](https://full-duplex-bench.github.io/) | source_or_metadata / retrieved |
| fdb-v3 | [原始来源](https://raw.githubusercontent.com/DanielLin94144/Full-Duplex-Bench/main/v3/README.md) | source_or_metadata / retrieved |
| fsc-paper | [原始来源](https://arxiv.org/abs/1904.03670) | abstract / retrieved |
| fsc-public-license | [原始来源](https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf) | source_or_metadata / retrieved |
| fsc-repo | [原始来源](https://raw.githubusercontent.com/lorenlugosch/end-to-end-SLU/master/README.md) | source_or_metadata / retrieved |
| fsc | [原始来源](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/) | source_or_metadata / retrieved |
| full-duplex-v3 | [原始来源](https://arxiv.org/abs/2604.04847) | abstract / retrieved |
| fun-asr | [原始来源](https://raw.githubusercontent.com/QwenAudio/Fun-ASR/main/README.md) | source_or_metadata / retrieved |
| funasr | [原始来源](https://raw.githubusercontent.com/modelscope/FunASR/main/README.md) | source_or_metadata / retrieved |
| granite-speech | [原始来源](https://huggingface.co/ibm-granite/granite-speech-3.3-8b/raw/main/README.md) | source_or_metadata / retrieved |
| heartbeat | [原始来源](https://arxiv.org/abs/2610.26233) | abstract / retrieved |
| heysquad | [原始来源](https://huggingface.co/datasets/yijingwu/HeySQuAD_human/raw/main/README.md) | source_or_metadata / retrieved |
| hubert | [原始来源](https://arxiv.org/abs/2106.07447) | abstract / retrieved |
| jarvis-full | [原始来源](https://arxiv.org/html/2610.23804v1) | full_text / retrieved |
| jarvis-paper | [原始来源](https://arxiv.org/abs/2610.23804) | abstract / retrieved |
| jarvis-repo | [原始来源](https://raw.githubusercontent.com/JensenYX/Jarvis/main/README.md) | source_or_metadata / retrieved |
| jev-agentdecision-paper | [原始来源](https://arxiv.org/abs/2610.21412) | abstract / retrieved |
| jev-audit-full | [原始来源](https://arxiv.org/html/2609.32160v1) | full_text / retrieved |
| jev-doc-intro | [原始来源](https://docs.typesafe.ai/introduction) | not_read / proxy_block_page |
| jev-harness-full | [原始来源](https://arxiv.org/html/2610.21412v1) | full_text / retrieved |
| jev-harness | [原始来源](https://raw.githubusercontent.com/JevAgentHarness/JevAgentHarness/main/README.md) | source_or_metadata / retrieved |
| jev-inference-full | [原始来源](https://arxiv.org/html/2610.02076v1) | full_text / retrieved |
| jev-inference-paper | [原始来源](https://arxiv.org/abs/2610.02076) | abstract / retrieved |
| jev-intro | [原始来源](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | source_or_metadata / retrieved |
| jev-rules-full | [原始来源](https://arxiv.org/html/2609.26758v1) | full_text / retrieved |
| jev-rules-paper | [原始来源](https://arxiv.org/abs/2609.26758) | abstract / retrieved |
| jev-score | [原始来源](https://docs.typesafe.ai/primitives/score) | not_read / proxy_block_page |
| kev | [原始来源](https://raw.githubusercontent.com/jaredpalmer/kev/main/README.md) | source_or_metadata / retrieved |
| laya | [原始来源](https://raw.githubusercontent.com/NandhaKishorM/laya/main/README.md) | source_or_metadata / retrieved |
| livekit | [原始来源](https://raw.githubusercontent.com/livekit/agents/main/README.md) | source_or_metadata / retrieved |
| m3slu | [原始来源](https://arxiv.org/abs/2510.19358) | abstract / retrieved |
| macslu-card | [原始来源](https://huggingface.co/datasets/Gatsby1984/MAC_SLU/raw/main/README.md) | source_or_metadata / retrieved |
| macslu-full | [原始来源](https://arxiv.org/html/2512.01603v1) | full_text / retrieved |
| macslu-paper | [原始来源](https://arxiv.org/abs/2512.01603) | abstract / retrieved |
| macslu-repo | [原始来源](https://raw.githubusercontent.com/Gatsby-web/MAC_SLU/main/README.md) | source_or_metadata / retrieved |
| massive | [原始来源](https://raw.githubusercontent.com/alexa/massive/main/README.md) | source_or_metadata / retrieved |
| mcrbench-full | [原始来源](https://arxiv.org/html/2508.15407v1) | full_text / retrieved |
| mcrbench | [原始来源](https://arxiv.org/abs/2508.15407) | abstract / retrieved |
| meld | [原始来源](https://raw.githubusercontent.com/declare-lab/MELD/master/README.md) | source_or_metadata / retrieved |
| minds14 | [原始来源](https://huggingface.co/datasets/PolyAI/minds14/raw/main/README.md) | source_or_metadata / retrieved |
| minicpm-o | [原始来源](https://raw.githubusercontent.com/OpenBMB/MiniCPM-o/main/README.md) | source_or_metadata / retrieved |
| mmar-card | [原始来源](https://huggingface.co/datasets/BoJack/MMAR/raw/main/README.md) | source_or_metadata / retrieved |
| mmar-repo | [原始来源](https://raw.githubusercontent.com/ddlBoJack/MMAR/main/README.md) | source_or_metadata / retrieved |
| mmar | [原始来源](https://api.github.com/search/repositories?q=MMAR+audio&per_page=5) | source_or_metadata / retrieved |
| mmau | [原始来源](https://raw.githubusercontent.com/Sakshi113/mmau/main/README.md) | source_or_metadata / retrieved |
| mmslu | [原始来源](https://arxiv.org/abs/2305.13090) | abstract / retrieved |
| mmsu-card | [原始来源](https://huggingface.co/datasets/ddwang2000/MMSU/raw/main/README.md) | source_or_metadata / retrieved |
| mmsu-paper | [原始来源](https://arxiv.org/abs/2506.04779) | abstract / retrieved |
| mmsu-repo | [原始来源](https://raw.githubusercontent.com/dingdongwang/mmsu/main/README.md) | source_or_metadata / retrieved |
| mmsu | [原始来源](https://api.github.com/search/repositories?q=MMSU+speech&per_page=5) | source_or_metadata / retrieved |
| moshi | [原始来源](https://raw.githubusercontent.com/kyutai-labs/moshi/main/README.md) | source_or_metadata / retrieved |
| nimble-paper | [原始来源](https://arxiv.org/abs/2609.32160) | abstract / retrieved |
| nimble | [原始来源](https://raw.githubusercontent.com/bespokelabsai/nimble/main/README.md) | source_or_metadata / retrieved |
| onejev-head | [原始来源](https://api.github.com/repos/OmniJev/OneJev/commits/main) | source_or_metadata / retrieved |
| onejev | [原始来源](https://raw.githubusercontent.com/OmniJev/OneJev/main/README.md) | source_or_metadata / retrieved |
| parakeet | [原始来源](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/README.md) | source_or_metadata / retrieved |
| pipecat | [原始来源](https://raw.githubusercontent.com/pipecat-ai/pipecat/main/README.md) | source_or_metadata / retrieved |
| prosodia-head | [原始来源](https://api.github.com/repos/alperiox/audio-jevlike/commits/main) | source_or_metadata / retrieved |
| prosodia-tree | [原始来源](https://api.github.com/repos/alperiox/audio-jevlike/git/trees/main?recursive=1) | source_or_metadata / retrieved |
| prosodia | [原始来源](https://raw.githubusercontent.com/alperiox/audio-jevlike/main/README.md) | source_or_metadata / retrieved |
| qwen-aligner-meta | [原始来源](https://huggingface.co/api/models/Qwen/Qwen3-ForcedAligner-0.6B) | source_or_metadata / retrieved |
| qwen-asr-06-card | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/README.md) | source_or_metadata / retrieved |
| qwen-asr-06-config | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/config.json) | source_or_metadata / retrieved |
| qwen-asr-17-card | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/README.md) | source_or_metadata / retrieved |
| qwen-asr-17-config | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/config.json) | source_or_metadata / retrieved |
| qwen-asr-full | [原始来源](https://arxiv.org/html/2601.21337v1) | full_text / retrieved |
| qwen-asr-head | [原始来源](https://api.github.com/repos/QwenLM/Qwen3-ASR/commits/main) | source_or_metadata / retrieved |
| qwen-asr-hf-card | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/raw/main/README.md) | source_or_metadata / retrieved |
| qwen-asr-hf-config | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/raw/main/config.json) | source_or_metadata / retrieved |
| qwen-asr-modeling | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py) | source_or_metadata / retrieved |
| qwen-asr-models | [原始来源](https://huggingface.co/api/models?author=Qwen&search=ASR&limit=100&full=true) | source_or_metadata / retrieved |
| qwen-asr-package | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/pyproject.toml) | source_or_metadata / retrieved |
| qwen-asr-paper | [原始来源](https://arxiv.org/abs/2601.21337) | abstract / retrieved |
| qwen-asr-preprocessor | [原始来源](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/raw/main/preprocessor_config.json) | source_or_metadata / retrieved |
| qwen-asr-processing | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/qwen_asr/core/transformers_backend/processing_qwen3_asr.py) | source_or_metadata / retrieved |
| qwen-asr-repo | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/README.md) | source_or_metadata / retrieved |
| qwen-asr-train-code | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/qwen3_asr_sft.py) | source_or_metadata / retrieved |
| qwen-asr-training | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-ASR/main/finetuning/README.md) | source_or_metadata / retrieved |
| qwen-asr-tree | [原始来源](https://api.github.com/repos/QwenLM/Qwen3-ASR/git/trees/main?recursive=1) | source_or_metadata / retrieved |
| qwen-audio-agent | [原始来源](https://raw.githubusercontent.com/QwenAudio/qwen-audio-agent/main/README.md) | source_or_metadata / retrieved |
| qwen-audio-jev-full | [原始来源](https://arxiv.org/html/2610.21614v2) | full_text / retrieved |
| qwen-audio-jev-hf | [原始来源](https://huggingface.co/api/models?search=Qwen-Audio-Jev&limit=30&full=true) | source_or_metadata / retrieved |
| qwen-audio-jev-paper | [原始来源](https://arxiv.org/abs/2610.21614) | abstract / retrieved |
| qwen-audio-jev-search | [原始来源](https://api.github.com/search/repositories?q=Qwen-Audio-Jev&per_page=10) | source_or_metadata / retrieved |
| qwen-audio-models | [原始来源](https://huggingface.co/api/models?author=Qwen&search=Audio&limit=100&full=true) | source_or_metadata / retrieved |
| qwen-audio-org | [原始来源](https://api.github.com/orgs/QwenAudio/repos?per_page=100) | source_or_metadata / retrieved |
| qwen-funasr-models | [原始来源](https://huggingface.co/api/models?author=FunAudioLLM&search=ASR&limit=50&full=true) | source_or_metadata / retrieved |
| qwen-new-asr-hf | [原始来源](https://huggingface.co/api/models?search=Qwen-Audio-3.1&limit=50&full=true) | source_or_metadata / retrieved |
| qwen-new-asr-search | [原始来源](https://api.github.com/search/repositories?q=Qwen-Audio-3.1-ASR&per_page=10) | source_or_metadata / retrieved |
| qwen2-audio | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen2-Audio/main/README.md) | source_or_metadata / retrieved |
| qwen3-omni-models | [原始来源](https://huggingface.co/api/models?author=Qwen&search=Omni&limit=100&full=true) | source_or_metadata / retrieved |
| qwen3-omni | [原始来源](https://raw.githubusercontent.com/QwenLM/Qwen3-Omni/main/README.md) | source_or_metadata / retrieved |
| relisten-paper | [原始来源](https://arxiv.org/abs/2611.04480) | abstract / retrieved |
| repo-aliyun-yunqi-insight | [原始来源](https://raw.githubusercontent.com/Xudream/aliyun-yunqi-insight/main/README.md) | source_or_metadata / retrieved |
| repo-tokrouter-examples | [原始来源](https://raw.githubusercontent.com/gatellm-io/tokrouter-examples/main/README.md) | source_or_metadata / retrieved |
| repo-tongflow-api-qwen-audio | [原始来源](https://raw.githubusercontent.com/tong-io/tongflow-api-qwen-audio/main/README.md) | source_or_metadata / retrieved |
| salmonn | [原始来源](https://raw.githubusercontent.com/bytedance/SALMONN/main/README.md) | source_or_metadata / retrieved |
| selectivenet | [原始来源](https://arxiv.org/abs/1901.09192) | abstract / retrieved |
| sem-dist | [原始来源](https://arxiv.org/abs/2104.02138) | abstract / retrieved |
| sensevoice | [原始来源](https://raw.githubusercontent.com/FunAudioLLM/SenseVoice/main/README.md) | source_or_metadata / retrieved |
| sfc-data-guide | [原始来源](https://raw.githubusercontent.com/QwenAudio/FunResearch/main/SpokenFC/data/README.md) | source_or_metadata / retrieved |
| sfc-data-manifest | [原始来源](https://raw.githubusercontent.com/QwenAudio/FunResearch/main/SpokenFC/data/manifest.json) | source_or_metadata / retrieved |
| sfc-full | [原始来源](https://arxiv.org/html/2608.05126v2) | full_text / retrieved |
| sfc-head | [原始来源](https://api.github.com/repos/QwenAudio/FunResearch/commits/main) | source_or_metadata / retrieved |
| sfc-paper | [原始来源](https://arxiv.org/abs/2608.05126) | abstract / retrieved |
| sfc-repo | [原始来源](https://raw.githubusercontent.com/QwenAudio/FunResearch/main/SpokenFC/README.md) | source_or_metadata / retrieved |
| sfc-tree | [原始来源](https://api.github.com/repos/QwenAudio/FunResearch/git/trees/main?recursive=1) | source_or_metadata / retrieved |
| slam-slu-full | [原始来源](https://arxiv.org/html/2610.20844v1) | full_text / retrieved |
| slam-slu-paper | [原始来源](https://arxiv.org/abs/2610.20844) | abstract / retrieved |
| slu-llm | [原始来源](https://arxiv.org/abs/2311.07418) | abstract / retrieved |
| slu-recent | [原始来源](https://export.arxiv.org/api/query?search_query=%28all%3A%22spoken+language+understanding%22+AND+%28all%3A%22large+language%22+OR+all%3A%22instruction%22%29%29&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending) | source_or_metadata / retrieved |
| slue-card | [原始来源](https://huggingface.co/datasets/asapp/slue/raw/main/README.md) | not_read / request_failed |
| slue-hf-meta | [原始来源](https://huggingface.co/api/datasets/asapp/slue-phase-2) | source_or_metadata / retrieved |
| slue-paper | [原始来源](https://arxiv.org/abs/2111.10367) | abstract / retrieved |
| slue-phase2 | [原始来源](https://huggingface.co/datasets/asapp/slue-phase-2/raw/main/README.md) | source_or_metadata / retrieved |
| slue-toolkit | [原始来源](https://raw.githubusercontent.com/asappresearch/slue-toolkit/main/README.md) | source_or_metadata / retrieved |
| slue | [原始来源](https://raw.githubusercontent.com/asappresearch/slue/master/README.md) | not_read / request_failed |
| slue2-paper | [原始来源](https://arxiv.org/abs/2212.10525) | abstract / retrieved |
| slurp-audio-license | [原始来源](https://zenodo.org/api/records/4274930/files/LICENSE.txt/content) | source_or_metadata / retrieved |
| slurp-head | [原始来源](https://api.github.com/repos/pswietojanski/slurp/commits/master) | source_or_metadata / retrieved |
| slurp-hf-meta | [原始来源](https://huggingface.co/api/datasets/qmeeus/slurp) | source_or_metadata / retrieved |
| slurp-license | [原始来源](https://raw.githubusercontent.com/pswietojanski/slurp/master/LICENSE) | not_read / request_failed |
| slurp-manifest-devel | [原始来源](https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/devel.jsonl) | source_or_metadata / retrieved |
| slurp-manifest-test | [原始来源](https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/test.jsonl) | source_or_metadata / retrieved |
| slurp-manifest-train | [原始来源](https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/train.jsonl) | source_or_metadata / retrieved |
| slurp-paper | [原始来源](https://aclanthology.org/2020.emnlp-main.588/) | source_or_metadata / retrieved |
| slurp-zenodo | [原始来源](https://zenodo.org/api/records/4274930) | source_or_metadata / retrieved |
| slurp | [原始来源](https://raw.githubusercontent.com/pswietojanski/slurp/master/README.md) | source_or_metadata / retrieved |
| smart-turn | [原始来源](https://raw.githubusercontent.com/pipecat-ai/smart-turn/main/README.md) | source_or_metadata / retrieved |
| snips | [原始来源](https://arxiv.org/abs/1805.10190) | abstract / retrieved |
| speech-jev-paper-search | [原始来源](https://export.arxiv.org/api/query?search_query=all%3AJev%20AND%20%28all%3Aspeech%20OR%20all%3Aaudio%29&start=0&max_results=30&sortBy=submittedDate&sortOrder=descending) | source_or_metadata / retrieved |
| speech-jev-search | [原始来源](https://api.github.com/search/repositories?q=jev+speech&per_page=15) | source_or_metadata / retrieved |
| speech-tool-recent | [原始来源](https://export.arxiv.org/api/query?search_query=all%3A%22speech%22+AND+%28all%3A%22tool+calling%22+OR+all%3A%22tool+use%22%29&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending) | source_or_metadata / retrieved |
| speechbrain-slurp | [原始来源](https://raw.githubusercontent.com/speechbrain/speechbrain/develop/recipes/SLURP/README.md) | source_or_metadata / retrieved |
| speechgpt | [原始来源](https://arxiv.org/abs/2305.11000) | abstract / retrieved |
| speechrag | [原始来源](https://arxiv.org/abs/2412.16500) | abstract / retrieved |
| spokenwoz-site | [原始来源](https://spokenwoz.github.io/SpokenWOZ-github.io/) | source_or_metadata / retrieved |
| spokenwoz | [原始来源](https://raw.githubusercontent.com/AlibabaResearch/DAMO-ConvAI/main/spokenwoz/README.md) | source_or_metadata / retrieved |
| superb | [原始来源](https://arxiv.org/abs/2105.01051) | abstract / retrieved |
| tau-voice-full | [原始来源](https://arxiv.org/html/2603.13686v1) | full_text / retrieved |
| tau-voice | [原始来源](https://arxiv.org/abs/2603.13686) | abstract / retrieved |
| tau2-head | [原始来源](https://api.github.com/repos/sierra-research/tau2-bench/commits/main) | source_or_metadata / retrieved |
| tau2-voice | [原始来源](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/src/tau2/voice/README.md) | source_or_metadata / retrieved |
| tau2 | [原始来源](https://raw.githubusercontent.com/sierra-research/tau2-bench/main/README.md) | source_or_metadata / retrieved |
| tev1 | [原始来源](https://raw.githubusercontent.com/togethercomputer/tev1/main/README.md) | source_or_metadata / retrieved |
| text-to-voice | [原始来源](https://arxiv.org/abs/2605.15104) | abstract / retrieved |
| ultravox | [原始来源](https://raw.githubusercontent.com/fixie-ai/ultravox/main/README.md) | source_or_metadata / retrieved |
| unисlu | [原始来源](https://arxiv.org/abs/2507.12951) | abstract / retrieved |
| voiceagentbench | [原始来源](https://huggingface.co/datasets/krutrim-ai-labs/VoiceAgentBench/raw/main/README.md) | source_or_metadata / retrieved |
| voicebench-card | [原始来源](https://huggingface.co/datasets/hlt-lab/voicebench/raw/main/README.md) | source_or_metadata / retrieved |
| voicebench-meta | [原始来源](https://huggingface.co/api/datasets/hlt-lab/voicebench) | source_or_metadata / retrieved |
| voicebench | [原始来源](https://raw.githubusercontent.com/MatthewCYM/VoiceBench/main/README.md) | source_or_metadata / retrieved |
| voicechat-card | [原始来源](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/raw/main/README.md) | source_or_metadata / retrieved |
| voicechat-full | [原始来源](https://arxiv.org/html/2609.21967v2) | full_text / retrieved |
| wav2vec2 | [原始来源](https://arxiv.org/abs/2006.11477) | abstract / retrieved |
| wavlm | [原始来源](https://arxiv.org/abs/2110.13900) | abstract / retrieved |

## FSC 下载补充核验

用户指定的 [OpenDataLab FSC 镜像](https://opendatalab.com/OpenDataLab/Fluent_Speech_Commands/tree/main) 通过官方 OpenXLab API 取得文件清单（dataset ID 1505）。网页入口在当前网络返回风险提醒页；公开 API/文件解析接口可匿名访问，无需 HF 或 OpenDataLab 登录。归档以发布 SHA256 固定；API 未提供不可变 commit ID。

README/metafile 的目录 size 分别为 1,017/434，但实际为 1,281/698 字节；实际内容 SHA256 均与目录给值一致，保留该差异，不修改原文件。重复归档和 `.aria2` 下载状态未入库。原始发布者专用许可与镜像 CC BY-NC 4.0 标签均保留。
