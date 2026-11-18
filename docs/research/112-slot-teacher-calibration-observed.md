# 槽位教师校准终态：主要缺口是输出截断

距前次观察617秒后，1794525979.2778935检查PID3831451已不存在，89/89请求及
总终态齐全。没有重启或重试。原89请求/110目标固定包按严格完整响应规则评分。

89请求全部HTTP200、执行器exit0，但只有75请求以stop返回结构完整答案，
14请求均length且正文为空；每个失败请求4096输出token，4092–4096为reasoning。
因此HTTP/退出码成功不能当语义输出成功，不能从截断思考中恢复标签。
完整请求率75/89=84.27%，低于已登记98%门槛，当前profile不能全量派发。

| 面板 | 完整 / 全部 |
|---|---:|
| S00 | 12/12 |
| S01 | 12/13 |
| S02 | 12/13 |
| S03 | 9/12 |
| S04 | 7/13 |
| S05 | 12/14 |
| S06 | 11/12 |

110指定目标中94返回且与作者参考一致，16因请求无效缺失：正47/55、负47/55。
保留全部分母的参考一致率94/110=85.45%；返回条件下100%不能当独立准确率，
也不能掩盖失败。该包没有歧义参考目标，不能估计unavailable判决质量。
75完整请求共589判决，额外495判决只是未评分proposal，不自动成为gold。

失败目标涉及coffee_type、game_name/game_type、joke_type、list_name、music_album、
music_descriptor、music_genre、order_type两极、person/personal_info、player_setting、
radio_name、relation和transport_type。已逐项查看失败输入与返回状态；这些包含简单
表述，不能将失败解释成任务必然复杂。S04最差，S00本轮全完整，但不推断稳定上界。
另查看每个面板至少一项完整参考匹配的理由，不替代独立质量审核或全额外标签审核。

总输入1825744、输出185687、合计2011431 token；未据未提供的价格估算费用。
输出上限是预算约束而不是返回保证。历史8192同源探针仍失败，不自动扩大上限或
对失败做循环重试。下一步应验证更可控的教师输出方式或独立审核器，再冻结全量预算；
本次结果不支持直接启动128组质量队列或147072请求全量标注。

结果包：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/slot-teacher-calibration-observed-v1/`。
report SHA256 `51386feca8002f6229fb2802f372bb4cf5ad7abf7afeb4c80ec50cf46a0b62fd`。
[独立核验](slot-teacher-calibration-observed-check-v1.json)逐项核查实际wire、完整终态、
原始response、全部ID与状态、usage和评分分母，未调用scorer内部评分函数。
核验脚本`.local/research/slot-teacher-calibration-observed-v1/verify.py`。
scorer历史schema名称含intent，但报告内容明确为slot参考一致率。

semantic_admission仍0，没有语义分歧目标不意味着监督已认证；当前任务包、工程基线、
学术复现及RL曲线均未完成。本轮未训练，原教师进程已终态，无需继续轮询该PID。
