# 全意图教师参考校准

任务先行阶段，继续核验教师能否消费已设计语义规约；不启动模型训练、不把校准参考
当成正式监督、不改变65张意图卡或55张槽位卡。上轮完成工程收口核对以及8192输出
预算失败证据归档，本轮据此推进质量校准，不继续盲目增加槽位输出预算。

## 固定输入和选择

工具 `tools/jev/build_intent_teacher_calibration.py` 消费已校验文件hash的
intent-presence-inventory-v3和blind-semantic-annotation-shared-v1。
对每张意图卡按固定hash顺序选择1个正参考和1个负参考，加16个hash顺序的歧义参考。
重复评估保留全部来源证据，不重复计覆盖；同坐标冲突不可投票/取最新值解决。
不兼容规则直接拒绝，未决不能供应负例。

共享产物：
`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-teacher-calibration-v1/`

- 65卡，65正、65负、16未决，共146目标判决、96来源。
- 同来源同面板合并后122请求，全部返回时962判决；非目标答案仅存为新建议。
- requests.jsonl保留既有完整固定面板及规则，只有reference_text随来源变化。
- 预期状态、评估证据及原始source_id只在cases.offline.jsonl，绝不进入教师wire。
- 选择使用历史参考是离线校准分层，不是actor运行时按gold路由Q，也不是全人口指标。

独立核验直接读取原metadata的sentence，逐项重建146参考绑定、全部评估hash、完整面板
和字段allowlist，并确认每卡恰有一个正负目标。证据
`docs/research/intent-teacher-calibration-check-v1.json`，packet report SHA256：
`d9dcf46dc55ff5e29a34b7c4aa8452d74954a311e18239c0cd871de69d73167e`。
这些检查没有完成独立语义复核或音频复听。

## 执行与评分

已启动独立进程，launch记录在`.local/research/intent-teacher-calibration-v1/launch.json`，
PID3830498，启动时验证进程存在；本记录不承诺后续实时状态。
runner位于同目录run.py，run/plan.json固定122请求、并发2、单请求4096输出token、
累计输出上限499712；thinking enabled，60秒网络/180秒总截止，不自动重试。
复用Omni已pin执行器及已有私有连接身份，不复制研究提示词。仅请求文件保存凭据引用，
目录0700、消息与配置0600，不在公开报告输出端点/密钥。

各请求保存意图、消息/config hash、PID、响应及终态；总进程独立会话运行，维护心跳和
总终态。此次是有限校准，不是147072请求的全人口标注派发。输入token没有官方tokenizer
逐项计量；历史意图面板3053–4104仅是已观测范围，不能作为新请求的严格上界或费用估计。
按用户约定，状态检查间隔至少600秒；完成后检查原句/面板身份，再评分。

`tools/jev/score_intent_teacher_calibration.py` 要求全部请求终态，校验packet/config/
messages/实际wire身份，严格stop及完整JSON/单围栏、所有ID和合法状态。失败请求的
目标仍留在130个明确参考的总分母；另报有效返回分母、正负与未决结果。
未决回答不当作负例，额外面板回答不冒充已知gold。24项相关测试通过，覆盖选择重复/冲突、
极性缺失、失败分母和额外答案隔离；尚无本轮教师完成或质量结果声明。

终态评分命令：

```bash
.venv/bin/python tools/jev/score_intent_teacher_calibration.py \
  --packet /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-teacher-calibration-v1 \
  --run .local/research/intent-teacher-calibration-v1/run \
  --output /mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-teacher-calibration-observed-v1
```

参考由此前研究整理，尚非独立审定gold。因此所有比率必须称“参考一致率”，不称
教师真实准确率。下一步阅读全部分歧及分层一致样本，区分定义歧义、参考错误和教师
理解错误，再确定独立审阅/准入方式。禁止把教师自洽或与现有参考一致直接升级为监督。
新语义训练准入保持0，工程与训练派发保持关闭。
