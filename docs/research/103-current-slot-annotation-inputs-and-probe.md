# 当前槽位规则完整标注输入与单请求验证

本轮把slot包v14接入既有标注构建器，原builder/share代码不变。完整literal工作包
blind-semantic-annotation-workpack-v2与shared-v2均发布在共享slurp-jev-benchmark目录，
保留v1历史，不原位覆盖规则或教师请求。

全部9192来源、40513录音绑定不变；65I/55S共16面板、每轮147072请求及1103040判决。
逐来源原metadata、逐录音target绑定、全日程及定义投影独立重建通过。全部9个意图面板
与旧版完全一致，仅S00/S02/S04/S05/S06的ontology变化；问题、顺序、schema不变。
原意图校准继续有效，无须重发。证据：

- `blind-semantic-annotation-check-v2.json`：literal report SHA256
  `5e342480e2b348ee73e730a2f1c3e1913f9280311c2cb55c83b0a9d5933b5b77`。
- `shared-annotation-check-v2.json`：shared report SHA256
  `2044cd0c097fc6a0a52cebea575932340b6ba5033a4103ba50a531f9bdc67f5d`。
- `blind-semantic-annotation-delta-v2.json`：精确版本差异。

共享规则逐字段还原16面板，完整147072日程及字节计算通过。新完整一轮请求内容
literal23,225,430,696字节/shared7,389,958,656字节；共享前后只是结构等价，未证明模型
理解等价。加入边界使新内容更长，不能以此声称上下文压缩优化或NPU准入。
相关24测试通过。

## S00完整8问对照

`.local/research/teacher-provider-rules-v1/run/` 保存一请求意图、配置/消息hash、PID、
响应与终态，session64306已exit0。复用原来源12657，原8问题及服务参数不变，
仅替换user.ontology为shared-v2。max_tokens4096、thinking enabled、不重试。

实际HTTP200/stop，45.39秒；23013输入、2693输出，其中2170 reasoning；正文2323字符。
严格单围栏JSON解析、8个ID各一次、合法状态及非空理由均通过。business_name和
business_type为positive，其他六项negative。两项已有当前规则参考与返回一致。

`teacher-provider-rules-observed-v1.json`保留结果，`teacher-provider-rules-check-v1.json`
独立校验实际发送请求只有ontology改变、确属当前面板，并重解析全部答案和两项参考。
旧版4096/8192及4+4后半组失败均保留，不能覆盖为成功。

该来源用于发现问题及编写规则，新规约明确包含AA flight、book a taxi等构式，因此
不能把本次一致当作独立语义准确率、未见样本泛化或新规约的因果收益。单次完成也不
保证重复可靠性或其他来源不会耗尽预算；新输入反而比原版多1160输入token。
其余六项只有本次教师建议，没有借此新增独立gold。

## 当前状态和下一步

监督计划及teacher profile切换新工作包，旧版完整记录归档；本轮只发送上述1个请求，
没有派发全人口标注。新语义监督准入仍为0，工程/训练保持未准入。
下一步在约定检查间隔核验122请求意图校准终态并分析分歧；槽位质量需独立于规则作者
选择的示例，完成正负/歧义校准及审阅。当前单请求结果不能替代全量预算或审核方案。
