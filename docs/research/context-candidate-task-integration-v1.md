# 上下文概率分支的数据与任务接口

状态：输入/标签桥接与真实CPU processor检查，尚未正式训练。接口为
`context_candidate_task.py`，接同一个语音骨干与共享浅Transformer概率头。

## 两种不可混淆的标签语义

- `exclusive_event`：原benchmark在完整本体下指定的互斥事件。全候选逐项判断
  “按该标注规则，本音频归属是否为此候选”；SLURP官方clean意图可支持一正多负。
  这不等于音频里提到了哪些意图，不能给I_present补负标签。
- `candidate_applicability`：调用方显式提供的候选定义下是否适用。允许多个true、
  全false或null。每个ID必须有bool/null条目；null是缺标，不能解释为false或unknown。

输入编译只接音频、候选ID/定义、规则和允许历史；所有定义先于首个marker，不接受
gold、不按样本主题选Q。监督单独绑定完整request hash；候选换序/ID重绑需同步标签
绑定，但问题语义和共享参数不改变。跨数据集可复用编译器，仍需各数据集自己的
官方本体与语义规约，不能因为接口相同就声称迁移有效。

## 与已有基线比较

旧Final60继续作为音频能力/工程锚点。新event协议输出60个sigmoid prob，主事件
准确率可按argmax与原60类指标对照，但逐候选阈值集合指标和概率校准须单独报告。
不要将sigmoid向量归一化后冒称原适用概率，也不把互斥事件的one-hot监督当多意图
检测证据。head结构、token、读出位置和问题模板都改变，任何收益不是单因素对照。

`context-candidate-budget-cpu-v1.json`：完整60候选、中位/最长音频分别2834/3104
实际token，60marker均完整且在音频之后，均<=4096。本地重建完整输入协议与hash
通过；没有全人口processor、NPU显存或模型任务输出证明。

## 学习目标仍需冻结

一正59负意味着纯逐项平均会被负例主导。不能直接把这套数据接默认BCE就称完成
Jev训练。对比/排序约束与概率监督应分工：正负候选对比学习区分能力；可衰减的
proper-scoring辅助项提供绝对概率锚点。重加权和负采样会改变最优概率的统计语义，
须保留自然分布校准评测、记录采样/权重和原query归一化，不能靠训练平衡宣称校准。

主RL基线继续GRPO/RLOO clipped，二元动作接口已有CPU数值验证，但整集合exact
reward可能稀疏、逐位correctness又容易全负投机；正式候选奖励需先与返回全部、
返回空集、单正例oracle和当前基座控制比较，再固定。不存在可从当前工程接口自动
推导出的通用“最佳reward”。已有候选集合计分器可报告precision/recall、empty-set、
完整集合、Brier/NLL和返回数量；真实效果评测仍待训练策略。

## 运行准入缺口

全人口真实processor与班次长度预检、生产token注册/行训练方案、NPU分支及LoRA
更新/恢复、vLLM分支加载与切换、冻结loss/reward/比较预算、完整学习与评测，均
未由本次接口/CPU结果证明。正式TEST保持不参与优化，FSC迁移不以重新训练替代。

全人口桥接验证已完成：学习40513、选择9226条，分别一正59负，负例2390267/544334；
逐条完整候选与原事件一致，类别计数另从发布JSONL独立复算。见
`docs/engineering/context-candidate-population-v1.json`。选择集仅核对评测转换，不用于更新。

## 全量预检与NPU准入排程

完整49,739条音频CPU预检已启动，使用4个独立processor进程，保持官方发布全量
来源与角色隔离。每条记录音频hash、请求hash、输入张量hash、60个marker位置和
完整长度；超过4096只记录overflow，不截断或删样本。结果尚未收集，不能把此前
两条长度结果当全量通过。tokenizer仍是试验占位版本；生产注册改变须重验输入。

`probe_context_candidate_npu.py`及`launch_context_candidate_npu5.py --prepare-only`
已准备，无远程派发。单卡5仅在空闲时使用，当前全量Stage1及其完整评测优先。
探针使用完整60候选、中位/最长真实音频，记录前向/反向耗时、allocated/reserved、
全部LoRA/分支梯度和原始重复输出；梯度来自无标签的诊断泛函，不做optimizer更新。
是新HF eager路径，没有自动继承原Final60候选profile的vLLM一致性验收。

若单条路径通过，再验batch/padding、原生verl更新与FP32master恢复；vLLM需实际
实现新分支及processor/marker行加载，不能仅在HF恢复后宣称完成训推一致。
