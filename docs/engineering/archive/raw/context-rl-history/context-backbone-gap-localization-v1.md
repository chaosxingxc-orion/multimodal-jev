# 实际NPU概率差异定位

固定镜像、同一联合策略67260b78、两条真实音频，head与LoRA不更新。
899c34ce已exit0/nonOOM，完整采集HF/vLLM进入head前的全部hidden及head参数，
worker在真实NPU上用相同head重放HF hidden；输入和参数逐张量核对通过。
独立记录见`context-hidden-gap-v1.json`，完整产物见
`context-engine-hidden-audio-npu7-v1.json`。

| 输入token数 | backbone hidden相对L2差 | 决策位置hidden最大差 | 同HF hidden的head概率差 |
| --- | ---: | ---: | ---: |
| 2834 | 0.021636 | 1.75 | 0 |
| 3104 | 0.036973 | 2.25 | 0 |

HF从保存策略重载的概率与原参考完全一致，vLLM原生输出也复现上一探针的
0.008534/0.010416概率差。因此两请求上的差异位于head之前；并未定位到backbone
具体某层，也不证明全部输入上head均完全等价。

下一候选复用现有`span_runtime_profile`的HF顺序对齐RPC（norm、rotary、
音频/语言attention、分离投影及短音频padding），通过显式诊断配置启用，
不改变`FixedContextRollout`默认原生路径，不更改模型权重或准入阈值。
`context-engine-aligned-audio-npu7-v1`容器7e4d2b20已在175物理7卡启动；
完成后必须重新检查同权重/输入及原始概率差，不能继承旧Final60一致性结论。

完整人口预检77d49da5已exit0并独立复核全部49739条请求、人口、位置、长度账本：
learning40513、selection9226，60marker，无截断/overflow，最长3104token。
证据`context-population-preflight-cpu-v1.json`。仅证明processor输入可消费，
不证明NPU batch容量、RL训练效果或任务泛化。

效率候选：独立审核可选`independent_input_memoization=true`，仅在本审核进程内
复用相同音频内容与完整请求的实际重建长度，不信任训练缓存、不复用监督。
既有运行中的审核使用原快照且未修改；两步旧checkpoint的新版审核3f6c28c3
已exit0，19个状态/人口/排列字段与旧结果一致。该前缀256输入无重复，
memo_hits=0，因此尚未覆盖重复命中或验证全量加速。全量候选仅prepare-only，
不得以此前缀成绩替代原fullaudit。
