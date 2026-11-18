# 新概率头的全量训练输入

已发布本地独立日程`.local/decision-objective-research/context-full-learning-schedule-v1`，
manifest SHA256 `d9cd5832aa160df46a4c10a2051ff12c352d2a2609e2cffd6901267b28a814f6`。
证据`context-full-learning-schedule-v1.json`，研究配置已绑定该身份。

日程通过official-clean loader复核40513 learning录音、9181来源组，再与已有新context
全人口processor ledger逐条核对audio、source_group、完整request hash、60marker、
长度与4096预算。没有复用旧Final60的输入验收。预检ledger SHA为
`46e2b4504c577a2da5e83c893ad9c8bb3e608373178ea2d1feae77f4af0ace2a`。

两个epoch各317次更新，每条真实录音每epoch仅出现一次、权重1；尾部63槽位权重0。
DP4/global128/micro8/accum4；先按类别时间分散排列，再在global batch内按实际
processor token数及特征padding宽度分组。特征宽度不是有效帧数，不混称音频时长。
所查32步窗口为59–60类；不是严格批内均衡。labels只用于日程，不用于选题，
所有query均消费全部60定义。分布不变但批次成员改变，不声称优化轨迹等价。

`cached_context_batches.py`用一个CPU线程提前处理下一整批，只把音频/processor输入
放入缓存。监督、query权重、当前策略在消费时绑定，采样和参考分布由后续RL入口提供。
恢复从固定step开始，异常不跳过样本，尾批仍使用全局真实query分母。
9项缓存/日程相关测试通过；尚未接入全量NPU训练，不把模块存在当成已实现吞吐提升。

## CPU缓存实测

固定镜像CPU容器bcae8183已exit0；256条真实learning输入的独立processor结果、
冷缓存、热缓存、整批预取逐张量一致。本地再复核256缓存文件内容hash、绑定键和
参考输入，见`cached-context-inputs-cpu-v1.json`。使用2CPU、torch1线程：
冷构建124.66秒，热读12.04秒，随后整批预取1.81秒，均为256条总服务时间。
顺序执行存在文件系统缓存及并发负载影响；未随机交叉重复，也未并发NPU更新。
不能把热读到预取的比例当成可归因的预取加速，更不能替代NPU稳定吞吐。

## 多卡审核修正

四卡v2 c10552d8已exit1/nonOOM。越过rendezvous，执行到第一次native update后的
检查，失败为optimizer state缺少step；完整证据保留。旧检查假定所有128个原参数
在每rank均有Adam状态，未处理FSDP use_orig_params的零长度未拥有分片。
新增context_optimizer_audit仅允许空分片无状态，非空分片必须有完整FP32 moments
和精确step；6项相关测试通过，新增每rank逐参数layout便于证伪该归因。

v3容器330a9923已exit0/nonOOM并完整独立收集，见context-native-actor-dp-npu-v3.json。
256条真实学习音频、两次global128/micro8×4更新及恢复重放成功，各rank master/Adam
逐张量精确；128个完整BF16导出张量跨rank一致，LoRA与head均变化，冻结参数保持。
实际每rank未拥有的空分片数13/13/11/8，符合v2审核错误的定位。
三次执行各rank最大耗时19.91/16.97/17.02秒，峰值allocated38.32GiB；仅短工程窗口，
不代表稳定吞吐，也未包括全量输入准备/rollout。没有RL或selection收益结论。
资源锁随准确wait终态释放，旧评测启动器可继续。
