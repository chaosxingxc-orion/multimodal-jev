# 预训练 Thinker 的真实 verl wrapper 比较协议

工具 `probe_omni_pretrained_verl.py` 已实现，当前未派发完整模型执行。
执行前要求原 TRAIN6 全部前向和独立验收通过，并固定 acceptance、inspection 哈希。
调用方必须重新查询原精确容器并捕获 inspection；不能用未终止容器或部分结果。

## 比较内容

使用既有完整六条学习集探针，每条固定首个声明决策，仍只输入真实音频和任务。
响应固定为 `false` 加 EOS，与原 CPU 前向一致。标签文件不参与。
双音频回合依次与单音频回合配对，其余单音频补后，保留全部六条输入。
原始音频逐请求计算特征后补齐，不重新批量计算波形特征。

构造真实 DataProto，prompt 左 padding，固定 response 右侧对齐，无截断；
序列长度不得超过 4096。old_log_probs 全部 -2，仅作为接口占位，metadata 中
标记 synthetic_old_log_probs=true。不调用奖励模块，不把这些值写入采样记录。
该 batch 不是用于训练的完整 K 响应组，禁止作为 rollout 复用。

同一冻结 pretrained Thinker，CPU BF16/SDPA，真实 DataParallelPPOActor，
temperature=1、padded、SP=1、静态微批、无 fused、无 prefix grouping。
显式启用既有局部 CPU device/logprob 适配，Gloo world_size=1 仅满足 rank 查询。
分别执行 compute_log_prob 的 micro_batch_size=1 和 2。
与原已验收、逐条无 padding、FP32 log-softmax 读出的概率比较，并互相比较。

## 预设判定与证据

本 CPU BF16 诊断预设逐 token 最大绝对误差 <=0.1，rtol=0：
两种微批各自对原单条参考，以及两种微批之间，都必须满足。
该阈值包含 BF16 log-softmax 和批量矩阵计算的舍入误差，
不能用作 NPU/vLLM on-policy 容许误差或训练重要性比率的通过门槛。
保留每条原始概率和实际最大误差，超过阈值先保存比较结果再失败，不事后放宽。
正式 on-policy 训练仍需另行冻结更直接的 token/概率/比率检查。

读取模型前后均检查 checkpoint 文件哈希；加载报告不能有 Thinker 缺失或形状错误。
整个模型保持 requires_grad=false；不创建优化器。记录源码、输入、前置验收、
排序、完整模型文件身份、逐阶段耗时和峰值 RSS，原 old logprob/metadata 必须不变。
前置 CPU 验收不能代替当前结果的独立验收；该工具本身尚未配新的验收运行记录。

## 预检查与当前状态

`--preflight-only` 在前置验收成立后只校验输入、真实 processor 与构批，不加载权重。
现有容器测试覆盖实际 DataProto、混合音频特征拼接、causal 边界、padding 和超限拒绝。
只有前置 TRAIN6 验收和真实 processor 预检查通过，才登记下一次完整 CPU 模型运行，
不与当前 160 GiB 前向重复加载。预计 CPU 成本高于原六次逐条前向，需依据实际
双音频耗时再固定完整预算，不缩小样本集以满足时间。

通过只证明此固定模型/输入/CPU 路径的 wrapper 和微批概率兼容，
不证明任务质量、rollout 一致、NPU、分布式梯度、LoRA 或强化学习收益。
