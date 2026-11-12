# 常驻行为概率的训练输入绑定

2026-11-12 UTC。新增显式`resident_behavior`配置，连接宿主多策略验收凭证
与`intent_actor_inputs.prepare`。缺省路径保持原单策略行为；冻结训练快照不改。

## 两侧边界

宿主`bind_intent_resident_behavior.py`重新观察真实Docker完成并重算整个
resident suite验收，要求结果与提供的凭证完全相同。再核对当前训练adapter
的实际内容身份等于选定策略，生成新的输入配置与绑定记录；不修改原配置。

actor侧`intent_resident_behavior.authenticate`要求凭证本身以及全部证据文件
由训练config固定哈希，核对container/image、固定配置、人口、子策略位置
和policy身份。任何结果改变、缺失pin、选错策略或不完整凭证均拒绝。
隔离训练容器不查询Docker，真实退出由宿主绑定工具负责核验。

随后仍运行原prepare完整检查：gate终态、READY、基座与adapter身份、
HF/vLLM候选、音频重新编码、policy-bound请求摘要、每来源完整概率一致性。
监督标签和reward始终在音频模型输入之外，读取vLLM行为概率，不用actor
当前概率冒充rollout。

## 尚未完成的训练闭环

初版绑定输出只是输入消费配置，原恢复路径会拒绝不同acceptance_root。
后续开发版已增加[独立行为与恢复状态的联合认证](intent-resident-resume-v1.md)：
保留previous_root/previous_suite及其完整FP32 master/Adam验收，只允许经完整
resident_behavior认证的相同策略使用另一个行为概率目录。它不是可直接派发
的常驻训练计划，动态导出、在线调度及真实两步闭环仍待接入。
当前新函数没有真实常驻suite可供绑定，不能把模拟目录或旧独立引擎输出
转换成成功运行凭证。既有D/Q/P和utility冻结源码继续原路径。

40项本地测试通过：完整凭证消费、完整宿主复核、文件变化、错误策略/人口、
缺失pin和部分凭证拒绝，以及原来源日程和输入人口检查。测试的Docker和
模型身份使用受控替身；固定镜像验证亦不代表真实NPU行为消费成功。

同一组40项测试已在固定镜像的无设备CPU容器通过，实际容器退出码为0；
实现源码与测试快照哈希一致。[固定镜像验收凭证](intent-resident-behavior-cpu-v1-verified.json)
记录源码及日志身份。这一结果只验证绑定契约，不代表常驻NPU训练闭环完成。
该凭证对应初版冻结源码；后续恢复集成的独立验证见上方新版本文档。

关键实现：`src/multimodal_jev/intent_resident_behavior.py`及
`tools/jev/bind_intent_resident_behavior.py`。真实上线仍需多策略suite实际成功、
训练状态凭证接入及两个在线更新的独立验收。
