# 当前意图规约下的完整待重审队列收尾

本轮不改任何语义定义、教师模板或训练实现。读取完整165个needs_rereview坐标及
三张当前卡、请求/非请求和集合音乐共同政策后，逐项登记作者参考判决。覆盖163来源、
624录音、634来源录音绑定；47正、107负、11未决。相对历史参考数值变化0，但此次
逐项判断独立登记，未通过替换definition hash自动继承旧结论。

边界例：`provide contact location`对联系人修改为负，不代表查询对象已消歧；
`what a nice song`的正向音乐评价不因保存对象有歧义而失效；`if not then add`表达
条件性添加请求，不表示条件已成立；`save/store my contacts`继续保留备份与修改歧义。
原生类别仅作离线身份核对，未用于补出判决。

产物：

- 参考包：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-current-rule-rereview-v1/`。
- 合并清单：`intent-presence-inventory-v5/`，仍为40513录音/9192来源/65语义卡。
- [构建配置](../../configs/reviews/slurp-intent-presence-inventory-v5.json)与
  [独立结构核验](intent-presence-inventory-check-v5.json)。
- 作者显式ID判决表和构建脚本：`.local/research/intent-current-rule-rereview-v1/build.py`；
  独立核验脚本：同目录`verify.py`。报告保存脚本hash及输入身份。

清单1451评估/1266坐标/587来源；当前489正、658负、119未决、596214未评，
needs_rereview为0。181个不兼容历史评估完整保留，185个重复坐标保留多重来源，
不投票、不覆盖。65卡都有作者正负参考，不代表总体覆盖或独立质量认证。
report SHA256：`cb27fe84b7981a8847a7d485b46159b0f5835b61f23e8c2f5b1e513b89e41072`。

独立核验不调用构建器：从原metadata、learning/targets重建全部录音绑定，检查全部
历史评估多重集、卡/共同政策兼容、来源/录音状态和65卡统计；额外确认新增坐标恰为
原165项、其他坐标状态不变。此为结构和算术核验，不是另一名评审的语义复核。
没有训练器变更，未重跑NPU或泛化测试，semantic_admission仍0。

槽位校准在启动约626秒时检查：`/proc/3831451/cmdline`匹配原runner，40/89请求
终态、无总终态，进程仍活跃。检查时间1794524756.0638847，记录位于
`.local/research/slot-teacher-calibration-v1/observation-v1.json`。这是当时观察，
不声称当前完成，不重启、不部分评分；下一普通检查不得早于1794525356.0638847。

下一步收尾槽位校准并审核失败/分歧，定稿全量标注及独立质量审核协议；当前仍有
大量未评坐标和槽位历史参考重审，不能把完成本队列当作任务包冻结。任务监督与菜单、
CE/reward/评测配置共同发布后，再进入统一工程与算法验收。
