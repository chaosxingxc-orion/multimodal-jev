# SFC-Jev 完整选择集的候选提示比较协议

本协议在真实 SFC 教师候选生成、选择集推理之前固定选优规则。
当前仅实现比较器与合成验证，尚无真实候选名单、完整候选运行或选优结果。
权重保持冻结，actor 直接消费音频和任务定义。

## 固定人口与指标

使用 `sfc-jev-complete-shards-v1/selection.json` 的全部 18 个分片：
899 个原始会话、1106 个单回合 bundle、17469 个独立决策。
计划 SHA256 为 `239932c3cc568ee181a0d562e514b20d6341b2b7a4c83b6c22bbfaf2916d452c`。
TRAIN partition v3 SHA256 为
`84df8dc98723a50baa7352be11049cd8c692fb2e2da94ff5de650392abe09330`。
完整原始会话和共享音频的连接关系不能拆到 learning/selection 两侧。
这里的“会话成功”只表示数据集中所有给定回合的决策全对，
使用数据集给定历史，不代表真实交互中执行工具或自主生成历史后的任务成功。

所有候选与 parent 使用相同人口、模型、运行配置、代码和严格评分器。
按以下三个计数的字典序最大值选择：

1. 整个会话所有回合、所有决策完全正确的会话数。
2. 所有决策完全正确的单回合 bundle 数。
3. 还原的原生调用完全正确的 bundle 数。

候选的单回合全对数、原生调用全对数、格式合法决策数、schema 合法决策数，
任一低于 parent 就不入选。完全打平保留 parent；候选彼此打平按 p1、p2 顺序。
这里是完整选择集上的确定性选优规则，不是统计显著性或 TEST 提升结论。

使用原严格评分：不修复原始文本，不用 Noul 预测跳过 calls，
保持类型、NAN 字符串、重复调用及参数内部数组顺序的原有语义。
额外报告 active/inactive 函数三项全对计数，活动标签仅用于评分统计，
不决定哪些请求执行。报告 parent 与候选在会话和 bundle 层面修复/退化的配对数量。
不以大量 inactive 函数的正确率代替整段会话质量。

## 派发前登记与派发后比较

真实候选完成教师记录验证、独立语义审阅和完整 CPU context 验证后，
在任何选择集推理开始前写入 comparison spec，并在启动记录中保存其 SHA256。
名单只允许 `[parent,p1]`、`[parent,p2]` 或 `[parent,p1,p2]`；
只能根据学习反馈和工程资格确定名单，不能看部分选择集结果后删除候选。
spec 精确包含以下字段：

- `schema`: `sfc-jev-prompt-comparison-v1`。
- `plan`、`plan_sha256`: 固定完整 selection 计划。
- `roster`: 有序完整候选名单。
- `rules_files`: 比较 CLI 的 `rules_files()` 返回值，固定比较器及评分依赖源码哈希。
- `prompts`: 每个名字对应 `roster_path`、`instruction_sha256`、`candidate`。
  `candidate` 沿用候选执行配置的精确身份；parent 使用 null。
  `roster_path` 是预定的 completed-runs.json 路径，完成后的文件哈希写入比较结果。

比较命令为：

```bash
PYTHONPATH=src .venv/bin/python tools/jev/compare_sfc_prompt_selection.py \
  --spec /absolute/path/comparison-spec.json \
  --spec-hash REGISTERED_SHA256 \
  --output /absolute/path/new-comparison-result.json
```

比较器重新验收每个候选全部分片的精确 Docker 身份与原始预测，
检查跨候选模型、版本、代码和运行配置相同，允许提示及其 context 证据不同。
只有全部通过后才读取 selection 行的标签；为验证发布完整性会哈希整个 TRAIN
目标文件，但不解析 learning 标签，不读取 TEST 文件，不信任已有聚合分数。
任一候选分片缺失、失败或混入其他模型时停止比较，不能基于已完成部分选优。

源码哈希不一致会在读取标签前拒绝比较。时间顺序仍需启动记录证明，
比较器不会把自报时间戳或事后提供的 spec 哈希当成派发前登记证据。
源代码固定机制也不替代真实候选质量验证。

### 队列绑定修订（v2）

完整 selection 队列在 prepare 时必须传入 `--comparison-spec` 和
`--comparison-hash`，并在 queue-plan 中保存完整名单的绑定。
队列校验计划、实际 prompt/instruction/candidate 身份和预定 completed-runs 路径；
缺失或不匹配时在 dispatch-intent 写入前拒绝执行。后续每个分片也重新检查。
dispatch-intent 在第一次生成前落盘并 fsync，包含 comparison SHA256。
比较器要求 queue-plan 与 dispatch-intent 绑定同一份 spec，记录二者哈希。
这些是本地执行顺序的证据链，不是外部时间戳认证，不能防止有权限者整体伪造记录。
learning 和 TEST 队列不能误用 selection comparison 参数。

评分排序和数据人口没有改变。v1 规则哈希记录作为历史证据保留；
v2 登记新增的队列与比较器源码哈希，尚未生成真实候选或执行 selection。

## 后续范围

胜出结果必须另行绑定 TEST 的提示冻结协议后，才能运行候选 TEST；
当前候选执行接口仍拒绝 TEST。失败运行保留，不自动重放。
后续权重 RL、真实 rollout 概率一致性、跨任务泛化均需独立实验，
不能由本比较器测试或选择集结果推断完成。
