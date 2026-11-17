# FSC 完整提示对照的独立复核入口

冻结TEST队列本身已有六视图严格评分、逐录音修复/退化和McNemar诊断。
本次增加独立复核代码，不修改运行中的源码快照、提示、选择规则、解析合同
或统计指标。没有使用p1部分输出，没有新增推理或教师调用。

`src/multimodal_jev/fsc_independent_audit.py` 不调用生产 `parse_answer`、
`score_six_views` 或 `paired`：独立解析原始JSON，重算格式/schema合法性、
六视图正确数、四项配对正确与六视图全对，再按录音对齐parent/p1的四格表。
McNemar使用整数 `comb` 和精确有理比值后转float，与原算法独立核对；
极小值仍可能在最终float表示时下溢为0，不因此声称概率数学上为0。

分母为3793条录音，每条六视图，共22758次调用。六视图不合并为独立样本，
指标不新增置信门槛或多重比较推断。配对p值沿用冻结报告作为诊断，不能
单凭其大小声明跨任务泛化或RL有效。task release语义及运行来源仍由冻结
准入和源码审计认证；独立数值复算不替代任务定义验收。

命令入口 `tools/jev/audit_fsc_test_comparison.py` 要求：

1. 原完整队列成功终态存在，parent及全部登记提示均已完整结束。
2. 原 `test-comparison.json` 能由冻结准入/评分逻辑复算。
3. 实际 `docker inspect` 精确容器ID、镜像、退出状态、无OOM及原命令匹配。
4. 独立raw评分与每个child score及总comparison完全一致；配对四格计数
   精确一致，p值只容忍两种算法浮点求值差（rtol1e-9/atol1e-12）。

```bash
PYTHONPATH=src .venv/bin/python tools/jev/audit_fsc_test_comparison.py \
  --root /mnt/cephfs/xingchao/experiments/multimodal-jev/frozen-qwen3-omni-official-test-v2 \
  --output /path/to/new-independent-comparison-audit.json
```

输出独占创建。缺少全队列终态或任一结果失败时拒绝，不能用于输出部分
TEST成绩。工具保存实际inspect供复核；完整源码快照与资源/设备准入仍
须独立审核，不以当前入口的退出状态检查替代既有729份源码认证。

## 当前验证证据

相关测试43项通过，覆盖严格JSON、非法null与布尔/整数混淆、额外字段、
错误confidence、重复/缺失/绑定变化、原始响应与伪造缓存值区分、相同
平均正确率下的不同配对转移、全六视图交集、McNemar和部分队列拒绝。
测试命令：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_fsc_independent_audit.py tests/test_six_view_score.py \
  tests/test_direct_prompt_selection.py
```

已对认证的parent全量22758条原始响应实际复算；请求、目标、预测和result
SHA与原独立审计一致，所有metrics与原score完全相同。复算native正确1752，
六视图全对9，3793录音完整覆盖。原结果见
[parent正式报告](../../experiments/archive/raw/frozen-baselines/fsc-omni-official-test-v2-parent.md)。这只是
既有结果的独立复算，没有重新推理，也不构成p1提升或跨任务成绩。

p1尚在运行，完整比较CLI未运行。完成后先核验实际终态和不可变快照，
再使用该入口做独立数值复核，最后更新正式对照表。
