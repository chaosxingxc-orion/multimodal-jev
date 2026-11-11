# 可配置思考要求：直接音频接口 v1

2026-11-11。实现范围为要求条件化的一次判决接口，不生成文本思维链，不实现多轮控制器或隐状态迭代，不声称模型已理解要求。

`JevExample` 新增独立的 `thinking: ThinkingSpec`。配置包含版本、具体要求列表、执行模式、判决轮数预算和 `decision_only` 输出约束。
配置进入模型输入、导出 request hash 及 verl adapter 的同输入签名；SFT 显式 batch 调度可以保留它。
HF teacher-forcing 与 vLLM 共享音频提示渲染，思考要求位于官方 ASR 模板保留的 system 内容中；冻结直接音频入口和有限候选读出也显式渲染。
实际 processor/engine 的新配置推理、概率一致性和模型效果尚未运行，本次测试使用处理器夹具验证参数传递，不冒充真实推理。

配置示意（仅接口示例，尚非 SLURP 训练定义）：

```json
{
  "version": "evidence-v1",
  "requirements": ["Check negation scope using the complete audio."],
  "mode": "direct",
  "max_rounds": 1,
  "output": "decision_only"
}
```

严格校验非空要求、重复要求、整数预算及额外字段。文本推理输出模式被拒绝。
`bounded_review` 可序列化为待实现协议，但所有当前直接判决 renderer 都拒绝运行，防止请求三轮而悄悄只做一轮。
历史 transcript-evidence 入口拒绝带思考配置的新请求；配置化路线仅进入直接音频执行。
自由文本要求仍需数据发布审查：字段校验不能证明要求没有答案泄漏、语义冲突或无法监督的条件，不把这一接口当成监督质量验收。

旧数据缺省不添加 thinking 字段，既有提示保持原样。此兼容仅保留历史实验及无配置消融，新研究发布必须显式配置；目前尚无新 SLURP Jev 训练发布，所以没有通过缺省绕过该要求。
SFT/RL 源码身份新增 task_schema/thinking 哈希。新逻辑不应套用于旧运行的断点恢复；旧运行使用冻结快照，结果保持独立。

78 项相关测试通过：配置改变请求/提示、gold/参考转写不改变输入、HF/vLLM渲染入口一致、无声丢弃配置被拒绝、三类直接入口拒绝未实现复核、SFT调度与发布目标绑定、旧接口兼容，以及相关SLURP/rollout测试。
测试没有加载模型、调用教师、验证真实 verl 多步轨迹或训练效果。

后续：审查 clean SLURP 的任务与思考要求定义，按规则族/要求族构造受控变化，完成有质量门槛的SFT适配；再在固定镜像中验证真实processor输入、直接判决及逐步的复核控制器/策略学习。
研究契约见 [配置化思考方案](../research/10-configurable-thinking-decision-contract.md)。

## 固定镜像真实 processor 补充验证

已用一条clean TRAIN learning音频在固定镜像CPU运行官方Qwen3ASRProcessor，无权重、NPU设备或网络。
无配置、否定范围要求、修饰关系要求分别产生149/214/213个输入token；三种配置的音频特征及236个有效帧逐项相同。
要求在模板及tokenize/decode后完整保留，目标改变不影响prompt，teacher-forcing仅对6个合成响应token计loss。
该布尔响应只是输入管线夹具，绝非该录音的真实监督；没有打开目标文件。
vLLM request渲染与HF输入源一致，但未运行vLLM engine或检验实际模型概率/决策。

v1因源码快照缺少包初始化依赖而在import阶段退出1，失败证据保留。
v2补齐依赖并先做隔离import检查后，以新目录/容器执行，退出0，主体12.74秒。
独立核对实际容器身份、固定镜像、命令、2 CPU/8 GiB、无设备/网络、源文件/processor/结果SHA和终态。
见[验收记录](thinking-processor-v2-verified.json)。这推进了真实输入接口证据，尚不能证明模型遵循思考要求。
