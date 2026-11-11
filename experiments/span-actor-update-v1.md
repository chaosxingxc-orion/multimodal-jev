# BF16 Span actor 更新适配与准入

`span_actor_update.py`增加单rank完整request-group的更新接口。输入是vLLM pooling候选概率、输入身份、policy版本及完整Span动作，不使用生成式token responses。记录行为分布时detach/clone；更新前检查policy/request一致、行为joint logprob未被改写，以及每条输入全部候选/完整动作概率gate。失败发生在zero_grad/backward/step之前。

准入通过后调用固定verl的GRPO outcome advantage和vanilla clipped PPO，joint logprob按整个Span元组求和，以长度1动作映射损失；BF16执行参数通过FP32 master AdamW更新。任务奖励由独立审核scorer提供，接口不读取文本gold、不伪造过程监督。完整训练仍需调用方认证数据/输入/adapter身份，维护barrier、不可变版本、导出/重载和防止重复消费。当前接口不是分布式verl trainer已集成的证明。

固定镜像CPU合成测试已通过：一次真实verl损失及BF16 optimizer更新，gradient norm=0.69140625；目标动作概率增加，behavior old logprobs保持不变，master/state均FP32。合成policy的准入误差0；它没有vLLM采样，没有语音输入，不能作为真实语音RL结果。凭证 `docs/engineering/span-actor-update-cpu-v1-verified.json`。

真实语音HF/vLLM目前仍失败，不能调用此接口绕过gate。下一里程碑为：真实候选分布通过多样本gate → 固定BF16初始化的一次verl更新 → 保存新adapter与优化器状态 → worker新版本加载及概率复核 → 等预算SFT/RL/辅助奖励对照。当前无真实语音RL参数更新。

## 单步生命周期入口

`tools/jev/train_span_rl_step.py --config CONFIG --output NEW_ROOT`提供顺序单卡入口，仅允许隔离物理NPU4、BF16、reviewed temporal release。选择seed17下第一个互补标签pair，三项任务每batch各1正/1负，32组完整Span动作。当前是平行任务，奖励显式选择三项正确率均值，不制造Final依赖或过程奖励。

执行顺序：HF参考子进程退出 → 两条真实音频vLLM各自候选概率与完整动作gate → 新actor子进程重新认证输入、复验gate并使用verl更新 → 保存不可变policy-1 adapter/master/Adam → 独立actor进程重载，要求logp误差为0 → 两条vLLM新adapter复核 → READY。任一前置gate失败都没有optimizer.step；后置失败保留checkpoint但不生成READY。`update-claimed.json`以独占创建阻止重复消费，异常恢复必须新建run并指定完整checkpoint。

`span_checkpoint.py`同目录pending→rename发布，绑定base hash、adapter文件hash、optimizer hash、参数顺序、步数。恢复要求先加载匹配BF16执行参数，再恢复FP32 master/moments。历史BF16 SFT初始化没有optimizer，首个RL使用新的Adam状态，不能称为旧SFT优化器续训。

CPU验证共30项通过，覆盖数学/奖励/输入身份/概率gate、断点续训等价性、篡改检测、乱序参数拒绝、更新前失败阻断和更新后失败不发布READY。生命周期测试的后端是fake，仅证明调度控制；真实固定镜像NPU入口已在 `slurp-span-verl-vllm-step-v1` 完成首次验收：参考actor成功，第一条vLLM rollout完成，max logp差0.0652434、联合比率[0.886845,1.036827]，argmax无变化但gate失败。入口按预期在更新前退出，update.json/policy-1/READY均未生成，optimizer_updates=0。第二条rollout和更新后路径因前置gate失败未执行。当前未完成分布式verl worker/Ray调度，未获得真实语音RL收益证据。

固定镜像真实PEFT/safetensors checkpoint测试另通过：合成CPU线性模型LoRA训练7步后保存，独立重载并继续9步，每一步BF16输出与FP32 master均与不中断训练逐位相同。凭证 `docs/engineering/span-checkpoint-cpu-v1-verified.json`。这是存储/恢复验证，不是语音/NPU更新成功。
