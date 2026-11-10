# SFC 原生完整 TEST 推理协议与实现状态

目标：在完整官方 test_id 2590条/1986会话和 test_ood 771条/593会话上，
测量冻结 Qwen3-Omni 的端到端函数调用表现，作为 SFC-Jev 的原生横向比较基线。
不减少样本，不改变官方 supplied-history 输入，不将此离线协议称为在线工具交互。
当前只有实现、输入验证与合成测试；没有派发完整 SFC TEST，也没有模型成绩。

## 已固定的实现

`src/multimodal_jev/sfc_native_full.py` 只允许完整官方两个 TEST 划分，
绑定 native release manifest 和 requests 文件哈希，并独立检查唯一ID、会话分组、
完整阶段和双音频前缀。真实全量检查见 `sfc-native-full-population-input-v1.json`。
输入读取 `*.requests.jsonl`，不读取对应 targets 或原始合并标签文件。

`tools/jev/infer_sfc_omni_full.py` 直接使用原生 `sfc_omni.vllm_request`，
保留官方首轮任务定义、后续用户补充和提供的 assistant 历史；不引入教师指令。
固定 BF16、TP2、seed17、greedy、上下文4096、输出1024、max_num_seqs=1，
audio上限2、图像/视频0、关闭prefix caching，与TRAIN6一致。
沿用本地 Qwen3-Omni 30B-A3B Instruct、已核验revision和全部文件哈希，
torch/torch-npu/Transformers/vLLM/vLLM-Ascend版本必须与已通过TRAIN探针一致。
推理前后校验模型文件；只允许冻结基座、无adapter、audio-direct路线。

每条请求验证音频文件哈希，使用完整波形和官方不截断processor，
核对有效帧覆盖、有限特征、实际输入token加1024输出预算。
不把feature_extractor.n_samples误认为Omni最大音频长度；超长输入不能静默截断或删除。
实际vLLM返回的全部prompt token IDs须与同条HF processor完全一致。
生成原文、token IDs、finish reason、帧数、输入/提示/token哈希和耗时逐条保留。
length输出作为模型结果保留并统计；格式错误不能触发重新生成或偷偷修复。

## TRAIN 前置与完整验收

全量入口要求指定已经独立验收的 native TRAIN6 根目录、acceptance哈希和精确容器ID。
本地检查绑定六轮/四会话验收、完整推理结果、评分文件、源码、模型和相同运行配置。
它不从容器内查询Docker；后续派发器仍须在启动前确认该精确TRAIN容器成功退出。
当前TRAIN6仍在既有依赖队列，因此尚不能创建有效的正式全量运行配置。

`tools/jev/verify_sfc_omni_full.py` 在推理结束后独立检查：

- TEST和TRAIN前置的精确容器ID、成功退出、镜像；TEST只允许175的4/5或6/7卡。
- 启动记录中固定的config哈希、完整源码文件哈希；模型/版本与TRAIN一致。
- 全部有序请求/输出一一对应、无遗漏/重复、输入哈希与音频哈希绑定。
- 每条token预算、引擎prompt一致性、帧覆盖记录和length统计。

这些检查完成后才读取该完整划分的targets，逐条执行lossless restore_record，
重建原始评分输入并调用既有 `evaluate_sfc.py`。报告官方评分和严格评分，
保留level细分及多轮完整会话正确率；不以合法JSON代替任务正确率。
评分器固定SHA256 `fadf9ffd6a446dbe4e95b5f86b9a8998f5bba88cd515cca50db253962a51bd96`。
无正确率验收门槛，非法输出正常计错；工程损坏或缺失输出不能报告完整基线。

## 下一步

先完成已排队的TRAIN6和Jev78实际引擎验证，再冻结全量运行快照、配置、
容器启动记录及评分依赖，按现有NPU队列安排原生ID/OOD完整推理。
运行器和验收器已实现。`watch_sfc_native_full.py` 已接入依赖观察与顺序派发：
先等待固定进程/启动记录对应的Jev TRAIN78成功，并重新执行其完整工程验收，
再解析native TRAIN6验收，固定每个TEST配置及源文件哈希。
仅使用6/7卡，依次完成ID和OOD，每个容器推理上限86400秒，观察86700秒。
依赖观察上限110000秒；临时观察错误只重查，失败或模糊派发均不自动重试。
每个split只有完整工程验收后才读取标签评分；ID成功后才派发OOD。
本文件是协议，不是启动记录；精确进程/队列证据另存。
随后将原生结果与同官方划分的Jev结果比较。教师只使用TRAIN学习反馈，
完整TEST的输出和标签均不参与提示优化、奖励设计调参或候选筛选。

## 已启动的等待队列

`sfc-native-qwen3-omni-full-test-v1` 已固定源码
`8310f638e436f193103deddbebff22a14074bfd4` 并启动本地调度器：
PID2763532，process_start288952416，队列SHA256
`e6af71aed1b76c1d90ee2979b4a4a632c555a5f7da21991413afc277ba35f8e8`。
唯一直接依赖为已排队的 `sfc-jev-qwen3-omni-train78-v1`，
其精确进程和四份准备/启动文件哈希均已固定；原生TRAIN6输入/配置/源码也另行绑定。
已核验进程存活、状态waiting_jev_train78；两个TEST子目录尚不存在，未派发TEST容器。
证据 `sfc-native-qwen3-omni-full-test-v1-queued.json`。
当前队列只推进冻结原生基线，不启动权重训练或教师API。
