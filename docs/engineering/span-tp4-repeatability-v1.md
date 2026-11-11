# TP4 同引擎重复性与trace影响

## 不启用额外确定性控制的实测

`span-tp4-repeatability-v1`在物理NPU4–7完成，退出0。同一ASR0.6B模型、
同一已验收四卡训练adapter、同一请求身份，在同一个vLLM引擎内依次执行：
无trace两次→trace一次→无trace两次，然后完整48条suite（首条构成第六次调用）。
HF-order TP4 profile与上轮相同。显式mm_processor_cache_gb=0，
每次使用不同音频缓存UUID但实际音频特征/token/问题完全相同。
trace中四rank均包含音频conv输入，确认记录请求实际执行了音频塔。

| 比较 | 最大logit差 | 最大logp差 |
| --- | ---: | ---: |
| trace之前两次无记录调用 | 0.75 | 0.1526606083 |
| trace之后两次无记录调用 | 0.75 | 0.0966765285 |
| 第一次无记录与trace调用 | 0.5625 | 0.0663998723 |

五次调用的全部10对比较均不相同，第六次无记录调用也与第一次不同。
该证据证明波动在开启trace之前已经存在；trace不是唯一来源。
它不证明trace完全无扰动，也不能将波动直接归因于通信或某个kernel。

完整48条对HF仅1条通过，最大logp误差0.3073542118，
联合动作概率比[0.5811116642,1.5048396382]，5个Span首选候选变化。
继续禁止TP4准入RL。概率验收阈值未改变。

## 实现与验证

`span_repeatability.py`固定调用顺序并比较原始logits与归一化概率；
单独检查logits可发现被softmax消去的整体平移。
`verify_span_repeatability.py`独立复算全部差异，校验输入身份、文件pins、
四rank专属trace路径及真实音频执行证据；不把通过证据复核当作重复性通过。
相关CPU测试20项通过（重复性4项、trace7项、topology9项）。

## 后续单因素实验

固定镜像的 `vllm_ascend/batch_invariant.py`（Apache-2.0）提供
HCCL_DETERMINISTIC=strict、LCCL_DETERMINISTIC=1控制。
下一实验只设置这两个环境变量，不启用整体batch-invariant模式，
不替换matmul/softmax/norm，不调整模型精度或概率阈值。
新增worker回执记录变量实际值，避免只凭启动参数声称生效。
这验证变量已传入worker，不独立证明底层通信库实际选择了何种算法。

证据：`span-tp4-repeatability-v1-launch.json`、`span-tp4-repeatability-v1-verified.json`。
目标仍包括学术基线复现、语音Jev任务发布、RL改进与跨任务泛化；本轮仅完成工程诊断。

## 确定性归约对照已完成

`span-tp4-repeatability-collectives-v1`退出0，四worker均回执
HCCL_DETERMINISTIC=strict、LCCL_DETERMINISTIC=1。
唯一计算设置干预是这两个环境变量；新增回执只读取变量，不改算子。

同一请求五次控制调用的全部10对原始logits比较误差为0，随后第六次suite调用也完全一致。
本实验范围内无trace/有trace输出一致，重复性问题得到控制。
尚未验证第二个独立引擎或其他输入的重复性，不声称跨引擎确定性。
设置为一组干预，没有分别消融HCCL与LCCL，不区分二者单独贡献。

HF概率一致性仍失败：0/48通过，最大logp误差0.3686215878，
联合动作概率比[0.5476210687,1.4935568442]，11个Span首选候选变化。
稳定性通过不代表HF一致性通过，不准入RL。

使用本次稳定trace与上一轮已证实输出不变的HF trace离线比较，
报告保存在实验根目录 `trace-comparison.json`。下一步在此确定性配置下继续定位
投影与归约的固定数值偏差；已验收TP1路径不受影响。

证据：`span-tp4-repeatability-collectives-v1-launch.json`、
`span-tp4-repeatability-collectives-v1-verified.json`。
