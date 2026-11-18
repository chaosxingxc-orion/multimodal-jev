# 四卡两阶段基线的完整 selection 评测

范围仅为 clean SLURP 单音频原生60意图成员判决。Stage1完整11点为
0、32、64、96、128、160、192、224、256、288、318。Stage2初始策略及固定reference
始终取318终点；selection最优点单独报告，不回写reference。当前没有本轮selection分数。

旧selection模板使用dual协议；本轮训练服务使用replica协议和bounded attention。
新评测直接复用本轮封存source、镜像、processor、head、LoRA和两服务配置，
使用`context_replica_selection_client`接入replica协议。不能仅替换socket或复用旧通信握手。
输入始终来自音频缓存及任务定义，不使用参考转写。

## 输入、数值检查与计分

完整9,226条selection使用既有独立缓存证据，重新绑定本轮训练的release。
读缓存时逐文件和逐张量核对身份；缓存只读，缺失不重算。候选顺序为原生60意图。
固定阈值0.5，继续使用既有独立scorer报告set-F1、top1、macro-F1、NLL、Brier等；
不因阶段结果改变阈值，不用selection提供训练reward或难例采样。

每个checkpoint在独立HF进程执行固定256条输入面板，再由两个单卡服务执行完整9226条。
面板仅按token/声学长度两轴的16个最短/最长样本和ID哈希补足，完全不按gold或预测选取。
HF进程必须退出后才启动两服务，始终在四卡总预算内；模型服务batch32、rollout32、macro128。
完整输出产生后重排面板记录，与同策略HF逐项比较logits和prob，要求最大差均0。
只有数值检查通过才输出分数；饱和概率相等不能掩盖logits差异。
这仍为有限面板一致性，不声明全量数值一致性。

独立审核复原原缓存tensor hash所绑定的marker位置，重放全部macro日志和rollout分工，
核对容器、镜像、只读挂载、源码、服务启动/退出与实际策略，再重新计分并比对报告。
任何阶段失败保留准确句柄/证据并停止，不自动重启、不输出成功分数。

## 固定入口

- `context_replica_selection_client --phase prepare`：固定输入面板，无NPU执行。
- `prepare_context_replica_selection --action prepare/audit`：独立核验Stage1完整训练来源，
  将一个已登记checkpoint绑定到当前runtime；audit在新临时目录重建配置、策略、adapter和脚本，
  比较全部payload，完成后只删除该次自行创建的重建目录。
- `context_replica_selection_operations --action validate/run/audit`：复用现有四卡租约、
  Docker精确wait/collect与传输审计。run成功后还须独立审核，终态写operations_root/finalization。

模板为`configs/portable/operations/context-replica-selection-client.json`、
`prepare-context-replica-selection.json`、`context-replica-selection-operations.json`。
本地宏分别为`REPLICA_SELECTION_CLIENT_CONFIG`、`PREPARE_REPLICA_SELECTION_CONFIG`和
`REPLICA_SELECTION_OPERATIONS_CONFIG`。封包声明source_plan及SHA256/source_finalization、
source_step、已封存parent、selection_inputs和新output/remote_root；执行声明selection_package、
package/manifest、host/lease/shared_root/driver_root、operations_root/tag和显式selection_dispatch_allowed。
共享配置不绑定个人路径。长运行仍需既有后台生命周期，不能在交互shell中悬挂整条曲线。

封包只在模型source树外增加有SHA256绑定的selection-client.py；服务及其source manifest逐字不变。
所有point共用同一固定脚本，只通过source_step、策略快照和输出目录变化，不按checkpoint复制实现。
11点自动串接及完整curve收集已适配当前replica协议，真实NPU评测尚未完成；
控制器CPU验收不是模型数值或效果准入。

## 当前证据（2026-11-17）

完整输入绑定已生成于`/srv/multimodal-jev-xingchao/operations/context-selection-input-binding-v0.0.38`；
population、候选顺序和原缓存receipt与历史独立证据逐项一致，本轮release为
`38e4273d9328e4f02db081315e46f97a17628b974be7ba175254882e95421482`。
面板位于`context-selection-panel-v0.0.38/panel.json`，SHA256
`d03b78fd2313c0b78185fd90142e0e4d33bb685524ee2f69a0692ce9e9110932`。
输入绑定不等于远端缓存当前字节核验；实际消费时仍必检。

客户端/缓存/封包61项CPU检查、生命周期及独立审核16项检查通过。
测试使用真实张量身份、macro日志、完整9226输入绑定、生成runner的实际子进程等待，
模型计算与部分外部前驱使用显式替身，不代表NPU验收或模型分数。

终点318真实封包已生成：
`/srv/multimodal-jev-xingchao/operations/context-infonce-selection-0318-v0.0.38-prepared`，
manifest SHA256 `26eafd3550684279e4acea54dc615c38c6df416a8bf55bf246e955208a105c69`。
本地配置`.local/config/context-infonce-selection-0318-v0.0.38.json`，
独立重建审核配置为同目录`context-infonce-selection-0318-audit-v0.0.38.json`。
独立重建审核已完成，终态为`replica_infonce_selection_package_audited_not_executed`，
318策略身份及全部payload与重建包一致。尚无NPU执行或成绩。

## 完整曲线固定串接（v0.0.40）

入口`tools.reports.context_replica_selection_sequence`，模板
`configs/portable/operations/context-replica-selection-sequence.json`读取
`${REPLICA_SELECTION_SEQUENCE_CONFIG}`。提供：

- `predecessor`：100步效率驱动/config/terminal的准确路径和driver/config SHA256；
  使用pidfd等待，成功后重新独立审核实际效率产物。
- `deployment`：继承同一host/lease/shared_root/driver_root及release_evidence；
  不增加设备，不丢弃legacy发布审核文件的只读映射。
- `point_template`：同一source_plan及hash、source_finalization、parent和selection_inputs。
- `points`：预登记11个step及各自package/remote_root/operations_root/tag；可选
  manifest_sha256表示复用已有封包，消费时仍独立审核，不覆盖已存在的目录。
- `output`及`selection_dispatch_allowed=true`：本地结果目录和明确的全曲线派发开关。

所有点按0、32、…、288、318串行执行。每点prepare或独立审核已有包，再调用共享
point.run执行容器wait、collect与独立审核；每点记录不可变completed-NNNN.json。
任一点失败停止，无自动重试，不调用模型调度。正常远程观察仍固定900秒。
本地重复执行拒绝覆盖，不将已完成prefix当完整曲线。

新collector `tools.reports.collect_context_replica_infonce_curve`通过
`configs/portable/operations/context-replica-infonce-curve.json`接收
`${REPLICA_INFONCE_CURVE_CONFIG}`，输入source_plan/source_finalization及
`evaluations[{step,audit_config,audit_config_sha256}]`、output。
逐点重审传输、容器、HF256 exact、完整9226输入与macro日志、native score，比较各点
相同runtime、service执行参数、固定客户端、候选/输入缓存及面板。最后复用既有summarize
和InfoNCE发布格式，报告所有点及退化，区分source_selected与固定318 reference角色。
没有11个点的真实证据就不会选择模型，不读取FSC、MINDS或TEST数据。

本地69项CPU回归通过，覆盖旧InfoNCE发布兼容、当前封包/实际生成runner、point生命周期、
完整曲线和序列。新增检查拒绝缺点、顺序/来源/策略/输入/客户端/服务变化、概率对齐失败、
不可信运行来源和错误资源绑定；模型计算、远端前驱及容器在序列测试中是显式替身。
最初序列测试发现不可变commit_json不能覆盖累计完成清单，已改为逐点不可变记录后通过。

实际配置`.local/config/context-replica-selection-sequence-v0.0.40.json`已验证；
等待v0.0.39效率前驱后执行，318点复用已审核v0.0.38包，其余点使用v0.0.40新目录。
后台驱动PID2321924，启动记录1794952452.1566179；启动后现场ps确认存活并等待准确前驱。
结果根`/srv/multimodal-jev-xingchao/operations/context-infonce-selection-curve-v0.0.40/`，
终态为该根下terminal.json；这是队列登记和进程核验，不是NPU派发或评测完成。
正式634步GRPO不由此控制器派发；独立[正式训练后继](context-cached-formal-successor-v1.md)
等待本曲线释放同一资源租约，并重新审核曲线和效率证据后才可派发。
