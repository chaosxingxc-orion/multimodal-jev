# 连续RL组批与固定参考分布

2026-11-14，候选组件 `tools/jev/context_online_batches.py`，尚未接入真实连续NPU训练。
为保持已准入Stage1的356个源码哈希不变，本轮新增代码位于独立Stage2工具目录；
后续在Stage2版本冻结时再整理为可复用模块。本轮没有改变排队训练的源码与配置。

## 已实现并验证的约束

- 每个更新使用当前engine重新计算行为概率；不跨更新缓存行为策略。
- engine调用只接收输入张量、marker位置和audio token ID，不接收监督标签。
- G16完整二元动作组在同一组概率上采样；不重复16次音频前向。
- 固定Stage1末策略作为KL reference；缓存键包含参考策略身份、完整输入张量身份和位置。
  engine如果切换reference身份则拒绝；同音频不同问句或输入特征不会误命中。
- 组大小16、GRPO默认/RLOO备选、KL系数.01，辅助ranking和binary NLL各.3→.1，
  在前80%更新按cosine衰减。634步中的零起点506达到.1，此后保持。
- 采样seed由基础seed、零起点update、完整request slot ID确定，恢复时可复现。
- 准备成功后必须确认一次actor更新及next policy ID，才能消费下一批；
  策略无变化的零梯度更新允许同ID。确认动作不验证导出权重，需外层真实activation证据。
- 缺标、旧策略、身份错误、重复slot、错误权重、0/1饱和概率均拒绝；
  部分执行失败后会话失效，禁止悄悄重用已消费批次。
- 尾部padding仍保留完整动作组但权重0，禁止整批没有真实query。

`tests/test_context_online_batches.py`连同既有rollout binding共19项CPU测试通过。
测试使用确定性假engine，覆盖跨更新刷新、参考固定、输入变更、恢复采样、
错误策略、缺标、饱和概率、cursor、padding和辅助系数；不代表真实vLLM/verl闭环或收益。

## 仍需完成

1. 常驻vLLM joint-policy激活：LoRA/head一起更新、请求排空、相同冻结marker与processor，
   切换失败需停服。当前FixedContextRollout只支持固定策略，不能冒充已实现热切换。
2. 与真实原生verl更新及FP32 master/Adam连续恢复连接；每micro比较行为概率，
   预先冻结可接受的数值边界，不能沿用诊断阈值作为正式训练准入。
3. 固定reference完整学习输入概率的持久缓存：当前组件只是内存复用，
   需要独立身份、恢复及文件校验，不能将参考概率写进纯processor输入缓存。
4. 逐更新导出/激活成本测量与稳定窗口，最终以完整选择集曲线和FSC冻结迁移验证收益。

## 输入准备现场

本轮到期检查：175远程13:17:17 UTC，原全量cache容器9822c31d仍running/nonOOM。
progress为1536/40513、535.09秒，无已有cache命中（首次构建）。
原driver与完整训练/selection接续保持。下一普通远程观察至少600秒后。
另启动CPU-only计时容器856faac5，16条独立cache冷/热路径cProfile，
只读借用原输入资产、独立输出，不修改原writer；实际计时结果待收集。


## CPU计时结果与缓存辅助构建

856faac5 exit0已收集：16条冷准备7.586秒，热读取0.209秒。
cProfile定位每条编码两次tokenizer.get_vocab，共32次4.190秒，占冷准备约55%。
这是整份15万词表从Rust到Python的重复物化，不能把该开销归因音频特征计算。

独立候选d2ef2b2d exit0已收集：冷6.964秒、热.205秒、词表快照候选2.312秒。
候选只在每个线程私有只读processor建立时读取一次词表，get_vocab每次返回dict副本；
保留marker与保留token检查，processor不允许动态增加token。
三条路径的全部输入/positions张量hash精确一致。候选编码16条累计.793秒；
此前基线16条5.250秒。运行顺序、初始化和并行主机负载存在混杂，不声称端到端加速倍数。
证据context-cache-vocabulary-profile-v1.json及.local中原始cProfile。

已启动CPU辅助容器5fa4bed4，exactwait session70906，使用4CPU、8GiB，
从完整学习人口尾部向前准备，原9822c31d仍从头向后处理。
原始源码/原writer/config均未修改；只对同一cache目录开放写权限，其他原容器卷只读。
同key文件通过既有原子link只发布一次，冲突时逐张量/receipt检查一致；
原流程仍逐条读取校验并发布40513条完整清单，独立collector继续负责全部校验。
辅助任务自身的清单不替代主清单，不改变原训练接续依赖。
配置/源码/launch证据位于.local/decision-objective-research/context-learning-cache-assist-v1-*。
辅助构建尚未终态，无全量吞吐结论，不删除原有工作或提前派发训练。
