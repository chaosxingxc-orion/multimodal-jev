# SLURP 时间角色训练先导

目的：在 clean-v7 学习数据上走通审核、不可变发布、直接音频 SFT；不以此替代完整学术基线、复杂决策链收益或跨任务泛化实验。

## 数据与审核

在 source-bundles-v1 的 learning 来源中，按 date/time/timeofday 原标注存在性组成八个分层；各层来源数依次为 000:7339、001:86、010:348、011:18、100:1133、101:54、110:224、111:21。每层按 SHA256(`temporal-review-v1:` + source_id) 排序取六个来源，再按原 request ID 字典序取一条录音。没有使用模型预测或 selection/TEST 选样。

48 个来源的 reference、原标签和请求组成私有审核包：
`/mnt/cephfs/xingchao/datasets/slurp-clean/clean-v7-a3cd74482c32b74f/temporal-review-v1/`。

Codex 逐条核对参考文本语义与三种任务定义，写入独立审核账本。不是人工听审，也没有将自动转写当作新金标；沿用原语料的音频—参考文本对应关系，发布时重新验证音频文件 SHA256。此证据层级是 reference-semantic review，不能声称已核实音频中的每个词。

两项局部判断保留质量 mask：一处已表达分钟偏移却未标注 time，记 conflict；一处当前时刻表达的类型边界未明确，记 unavailable。其余 142 项对照定义与原标注通过审核。不会把这两项变成 null/false 训练目标，也不原位修改原生标注。

[任务定义](../../configs/tasks/slurp/temporal/tasks-v1.json)明确：noon/tonight 等在该派生任务采用 timeofday 约定；时钟内的 AM/PM 不单独创造 timeofday；询问“几点”不等于提供时间值。三种角色允许共存。该约定是当前学习证据支持的任务版本，不声称完整官方本体。

[发布器](../../src/multimodal_jev/slurp_training_release.py)认证 source receipt、学习文件、任务和证据文件，逐决策审核后写出 train.jsonl、masks、账本及 manifest。部分状态不会从 actor 预测产生；审核证据也不进入 model_input。发布失败清除临时输出，已有版本不能覆盖。

发布位置：
`/mnt/cephfs/xingchao/datasets/slurp-jev-reviewed/48511c803e33936cfa787301cac940773ed4e80d90e346841498a59205f5f584`。

独立重新计算所有发布标签并验证录音 hash 和完整 mask 覆盖，见[验收凭证](slurp-temporal-training-release-v1-verified.json)。date 正负各 24；time 正 24/负 22；timeofday 正负各 24；48 个来源连接组。仅限此小型训练版本，不解除其他 SLURP 数据的质量限制。

## 训练配置

运行根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-temporal-npu-sft-v1`。

- 固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，新代码另以只读快照挂载。
- 本地 Qwen3-ASR-0.6B-hf，BF16，直接音频 + schema + thinking → JSON value，无文本转写中间模型。
- 175 主机物理 4 号卡，容器只挂载 davinci4；运行器验证进程仅可见一张 NPU，逻辑编号 npu:0。
- rank 4 / alpha 8 的语言模型 q_proj、v_proj LoRA，学习率 1e-4，seed 17。
- 142 个判断固定打乱后每批 6 项，末批 4 项，共 24 个更新，一个 epoch；900 秒超时上限，4 CPU/32 GiB，禁止网络。
- 数据、模型及代码只读挂载；独立实验目录存结果、配置、代码哈希和优化器断点。

本次属于工程 SFT。单次 loss 或训练集拟合不构成泛化收益；没有把这个平衡小样本当作原生 SLURP 分布，也未构造或验证多步规则族留出。SFT 训练正常还不能证明 verl/vLLM NPU RL 路径完成。

验证命令：`PYTHONPATH=src .venv/bin/pytest -q tests/test_slurp_training_release.py tests/test_slurp_training.py tests/test_sft.py tests/test_rl_preflight.py tests/test_rl_run.py`，61 passed；包括发布器篡改/越界/空审核拒绝与已有训练入口回归。

## 后续证据

先核实本次训练终态、步数、adapter 和实际设备，再固定独立推理及评测协议。完整基线仍需覆盖清理后任务分母；候选链收益应按数据协同指南比较单次、固定复核、规则与学习策略。不能通过继续扩大这批小样本训练步骤来替代独立验证。
