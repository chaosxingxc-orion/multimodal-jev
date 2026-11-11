# BF16 Span actor 更新适配与准入

`span_actor_update.py`增加单rank完整request-group的更新接口。输入是vLLM pooling候选概率、输入身份、policy版本及完整Span动作，不使用生成式token responses。记录行为分布时detach/clone；更新前检查policy/request一致、行为joint logprob未被改写，以及每条输入全部候选/完整动作概率gate。失败发生在zero_grad/backward/step之前。

准入通过后调用固定verl的GRPO outcome advantage和vanilla clipped PPO，joint logprob按整个Span元组求和，以长度1动作映射损失；BF16执行参数通过FP32 master AdamW更新。任务奖励由独立审核scorer提供，接口不读取文本gold、不伪造过程监督。完整训练仍需调用方认证数据/输入/adapter身份，维护barrier、不可变版本、导出/重载和防止重复消费。当前接口不是分布式verl trainer已集成的证明。

固定镜像CPU合成测试已通过：一次真实verl损失及BF16 optimizer更新，gradient norm=0.69140625；目标动作概率增加，behavior old logprobs保持不变，master/state均FP32。合成policy的准入误差0；它没有vLLM采样，没有语音输入，不能作为真实语音RL结果。凭证 `docs/engineering/span-actor-update-cpu-v1-verified.json`。

此前真实语音HF/vLLM失败；当前命名BF16运行profile已在48条真实语音全部通过gate，不能绕过每版本检查。下一里程碑为：真实候选分布通过多样本gate → 固定BF16初始化的一次verl更新 → 保存新adapter与优化器状态 → worker新版本加载及概率复核 → 等预算SFT/RL/辅助奖励对照。该阶段尚无真实语音 RL 参数更新；后续 step-v4 已通过单步生命周期，见下文。

## 单步生命周期入口

`tools/jev/train_span_rl_step.py --config CONFIG --output NEW_ROOT`提供顺序单卡入口，仅允许隔离物理NPU4、BF16、reviewed temporal release。初始运行选择 seed17 下第一个互补标签 pair；恢复运行通过显式 batch_index 消费下一个 pair，三项任务每batch各1正/1负，32组完整Span动作。当前是平行任务，奖励显式选择三项正确率均值，不制造Final依赖或过程奖励。

执行顺序：HF参考子进程退出 → 两条真实音频vLLM各自候选概率与完整动作gate → 新actor子进程重新认证输入、复验gate并使用verl更新 → 保存不可变policy-1 adapter/master/Adam → 独立actor进程重载，要求logp误差为0 → 两条vLLM新adapter复核 → READY。任一前置gate失败都没有optimizer.step；后置失败保留checkpoint但不生成READY。`update-claimed.json`以独占创建阻止重复消费，异常恢复必须新建run并指定完整checkpoint。

`span_checkpoint.py`同目录pending→rename发布，绑定base hash、adapter文件hash、optimizer hash、参数顺序、步数。恢复要求先加载匹配BF16执行参数，再恢复FP32 master/moments。历史BF16 SFT初始化没有optimizer，首个RL使用新的Adam状态，不能称为旧SFT优化器续训。

CPU验证共30项通过，覆盖数学/奖励/输入身份/概率gate、断点续训等价性、篡改检测、乱序参数拒绝、更新前失败阻断和更新后失败不发布READY。生命周期测试的后端是fake，仅证明调度控制；真实固定镜像NPU入口已在 `slurp-span-verl-vllm-step-v1` 完成首次验收：参考actor成功，第一条vLLM rollout完成，max logp差0.0652434、联合比率[0.886845,1.036827]，argmax无变化但gate失败。入口按预期在更新前退出，update.json/policy-1/READY均未生成，optimizer_updates=0。第二条rollout和更新后路径因前置gate失败未执行。当前未完成分布式verl worker/Ray调度，未获得真实语音RL收益证据。

固定镜像真实PEFT/safetensors checkpoint测试另通过：合成CPU线性模型LoRA训练7步后保存，独立重载并继续9步，每一步BF16输出与FP32 master均与不中断训练逐位相同。凭证 `docs/engineering/span-checkpoint-cpu-v1-verified.json`。这是存储/恢复验证，不是语音/NPU更新成功。

命名profile与48条验收后启动step-v2：两条更新前rollout全部通过（max误差均≤1.1920929e-07）。进入actor时，request/policy哈希逐条完全相同，但持久化JSON list与内存dataclass tuple直接比较触发身份拒绝；尚未生成behavior/update/checkpoint，0次更新。已把bind_span_request返回内容统一为同一JSON规范表示，哈希算法不变；新增round-trip测试仍验证真实字段变化被拒绝。新目录step-v3重新执行全流程，不复用已claim目录。

step-v3更新前两条概率及actor重算完全一致，完成1次真实verl更新，gradient norm=3.984375；保存BF16 adapter及FP32 master/Adam成功。但独立HF重载max logp差0.15266055，被门禁拦截，无READY、未运行新版本rollout。源码发现verl首次导入会通过npu_patch全局替换Qwen3 RMSNorm/MLP/rotary；此前在loss内部延迟导入，使更新前后计算图实现不一致。step-v4提前导入真实verl并恢复三个原始HF函数对象，保持已验收的运行配置，从同一初始adapter重新执行，v3 checkpoint仅作失败证据。

step-v4完整终态0/READY：提前初始化verl并恢复三个HF算子后，两条更新前概率误差0；真实更新gradient norm=3.984375；独立HF重载误差0；两条更新后vLLM max误差1.4901161e-08与5.9604645e-08。导出112个BF16执行参数，master/Adam状态FP32；checkpoint hash与optimizer完整性复核通过。这验证单步训练/导出/重载/rollout生命周期，不构成任务收益。新adapter另行全48条独立HF/vLLM复核。

新adapter全量后验 `slurp-bf16-postupdate-suite-v2` 终态0/READY：独立HF重载产生48条新参考，vLLM同engine顺序请求全部通过；max logp差1.1920929e-07，完整动作比率[0.9999998808,1.0000002086]，已独立枚举384种动作复核。单步更新后跨后端一致性验收完成；未测真实NPU优化器续训、多步稳定性、分布式或held-out收益。


## 优化器恢复与均衡游标

恢复入口 `initial_optimizer_checkpoint` 必须与 `initial_adapter` 指向同一 checkpoint。
保存时将学习数据 manifest、问题顺序、seed、整轮样本顺序 SHA256、当前及下一 batch_index
同时绑定到 optimizer.pt 和 CHECKPOINT.json；拒绝重复 pair、跳过 pair、数据或排序改变，轮末不自动回绕。
历史 step-v4 没有游标，迁移必须提供固定哈希的 legacy-selection.json，核对历史 balance、实际输入与 optimizer 哈希。
恢复完整 FP32 master/Adam 状态，并记录恢复前步数及各参数 Adam step；不会重新初始化 Adam 来冒充续训。

`slurp-span-verl-vllm-resume-v1` 在隔离 NPU4 上从已通过 48 条验收的 step-v4 恢复，
消费第二对互补录音，仍用 raw 音频、BF16 LoRA、32 次完整动作采样及三项平行正确率奖励。
其验收包括更新前概率 gate、真实 verl 更新、独立重载精确一致及更新后 vLLM gate。
运行结果待归档；成功后仍须对新版本进行全 48 条独立 HF/vLLM 验收。
单次恢复不等于 NPU 不间断训练等价性或分布式训练稳定性。
