# SLURP 发布依赖的部署校验

适用范围：简单SLURP任务的reference建库、有限GRPO验收、100步效率、正式训练和selection。
保留冻结数据身份；部署补齐依赖，不原位修改发布内容、标签或模型源码。

## 失败结论与证据

2026-11-17，reference v0.0.37容器
`ad505c893c1619d7ab4ae2d9eb1cdea20f0dd83a556523afdfc58bbe9ee24434`
于21:25:17Z启动，21:28:57Z exit1/nonOOM。收集日志表明，builder调用legacy release
loader时找不到`configs/reviews/slurp-intent-development-reference-audit-v1.json`，
在reference推理之前退出。生命周期报告的`actual isolated terminal runtime differs`
是非成功容器终态的外层拒绝；服务关闭时的BrokenPipe不是已定位的首发原因。

证据目录：`/srv/multimodal-jev-xingchao/operations/context-reference-job-v0.0.37/`，
具体日志为`jev-reference-v0-0-37-collected/supervision/actor.log`；
失败终态为`finalization/terminal.json`。有限验收及效率后继因前驱失败停止。
失败证据保留，当前无镜像损坏或模型计算故障依据。

根因：Stage1部署已有`release_evidence`只读映射，但reference部署漏传；有限验收、
效率后继也曾只继承host/lease/shared_root/driver_root，会进一步丢失该映射。
v1发布引用的外部审核文件属于其不可变身份，不能跳过校验或删除引用来通过运行。

## 固定工程处理

`tools.reports.context_release_deployment`统一实现：

- `inherited`复制共享部署字段及可选`release_evidence`，后继不重新拼接机器地址。
- `audit`验证发布manifest、publication-config及其每一项pin；根据实际Docker命令中
  最深的bind解析目标位置，要求只读且字节哈希一致，拒绝缺挂载、可写覆盖或文件篡改。
- v2自包含发布使用既有portable verifier，核对发布内相对路径及全部文件身份。
- reference、训练和selection的launch前调用审计；实际容器挂载仍由inspect独立审核，
  本地命令校验不冒充远端执行。

部署映射集中在`.local/config`的本地宏里；共享模板不绑定个人路径。运行/排队配置
不改写，修复使用独立v0.0.39部署；沿用同一Stage1终点、封存模型源码及已审核TRAIN面板。
有限验收依次执行prefix、真实尾批、恢复，全部通过后才能运行100步效率候选。
正式效果训练仍受效率与工程准入门禁，不由失败重试自动放行。

## 验证边界

相关生命周期及selection审核CPU回归146项通过；追加后继继承断言和portable发布验证后，
依赖/有限序列/效率后继69项通过。测试明确替代NPU执行，不能充当训练验收。
实际v1发布的13项外部pin已用修正后的Docker命令逐项核对，全部只读且哈希一致。
远端以只读方式逐项核对13项文件哈希也通过；证据为
`/srv/multimodal-jev-xingchao/operations/context-release-deployment-v0.0.39-preflight.json`。
最终依赖校验版本的reference/训练/selection生命周期60项CPU回归通过。
上述部署和CPU证据本身不能证明reference建库成功、GRPO准入、任务收益或迁移收益；
后续真实reference收集与独立审核证据见下节。

## 全量reference审核结果（2026-11-18北京时间）

v0.0.39的原驱动已发布`reference_collected_and_audited`成功终态。实际容器
`6db7a8ab40631122227005a8091f0d589a528d37fe859ef0960bfb78984be8a8`
于2026-11-17T23:14:56Z退出，exit0/nonOOM；随后完成原驱动收集与独立审核。
40,513条不同输入、317个macro的缓存输出均从完整日志和独立输入身份重建核验，
没有用容器退出码替代审核。

证据根：`/srv/multimodal-jev-xingchao/operations/context-reference-job-v0.0.39/finalization/`。

| 产物 | SHA256 / 身份 |
| --- | --- |
| `terminal.json` | `979224ff9ee46d52c981c5a640f5a32d8e93f8b834339f1051baab1ecda3547b` |
| `audit.json` | `541b19840bcd58272af43100fa24627be06202a530c3f3d25ed3c66512e5b09f` |
| `reference-output-audit.json` | `d1bf2bdee4fdac6e6d186a9709ade44d6f2856f7fd2d28855e91e0fc63457826` |
| reference dataset manifest | `5d16cc09a81baf3c04a27c783258b30067ddf9ea164f4b89e64fb74f1f4925cb` |
| 固定Stage1终点318策略 | `926e4d1316cefea8e39272b5153cbc7fb96880eb67b0da42896cd07ddc620095` |

准入边界：`reference_dataset_journal_audited`证明缓存完整性和来源，
`full_population_parity=false`、`training_dispatch_allowed=false`保持不变。
同终点TRAIN256面板的exact证据单独绑定，不外推为40,513条的HF/vLLM全人口对齐。
有限验收后继继续重审前驱并依次执行prefix、真实尾批、恢复；此时尚无其NPU执行结果，
100步效率、正式训练及源/迁移收益仍待。现有冻结配置、代码pins和队列均未修改。

## 修复版本交接

v0.0.39 reference包manifest为
`200a5b26011799c6150956cf7d764389d7254671417ba72e4e92c8e22f3106eb`，
模型源码manifest仍为`77d9bd0b4aefb9966b643157f9a8b37e47d7b1bab528392fbe9b4bbbe98d1efc`。
普通/恢复/正式及reference驱动补充56项CPU回归通过。
新部署宏为`.local/config/context-reference-job-operations-v0.0.39.json`，
reference后台驱动PID2320858（启动记录1794951951.9306583）；
有限验收宏`context-cached-acceptance-v0.0.39.json`，驱动PID2320886。
这些是本地驱动启动记录，不能据此称NPU已启动或缓存完成。
终态分别位于operations根下`context-reference-job-v0.0.39/finalization/terminal.json`
和`context-cached-acceptance-v0.0.39/terminal.json`。
效率后继宏为`context-cached-efficiency-successor-v0.0.39.json`，固定最多100更新；
通过准确pidfd等待和独立审核串接，不自动派发正式634步训练。
