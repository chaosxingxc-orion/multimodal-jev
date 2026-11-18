# 固定源策略的全量目标评测部署

状态：封包、部署、收集入口及测试完成，尚无正式源曲线可供真实封包或NPU执行。
本流程对FSC3793、MINDS8168和SLURP clean TEST12991使用同一macro推理基座。
SLURP TEST是源域留出评测，FSC/MINDS为跨数据集迁移，不将它们混为同一收益。

`collect_context_macro_source_curve.reconstruct`从原collector抽出只读重建接口，
保留原完整训练与11个selection点的容器、传输、模型、输入和评分检查。
`context_transfer_source_admission.admit`再次重建这些证据，将已发布的曲线、全部
分数与runtime证明逐项比对；根据固定source-only规则确定角色，并重新读取实际
LoRA/head策略及native adapter。任何目标输入都在此之后才打开。

`prepare_context_full_transfer.py`复制正式父包的运行基座，复制所选策略的adapter/
head及必要元数据，主模型与音频cache只读共享。将全量portable输入、parity输入和
源selection分数封装进本地新目录；完整源分数包可搬迁，独立ORIGIN仍绑定原证据。
对三个角色分别准备：common_Stage1、source_selected_new_GRPO、fixed_endpoint_new_GRPO。
如果多个角色policy ID相同，最终矩阵应复用同一份实际推理输出，不能据目标成绩选角。

`context_full_transfer_operations.py`支持validate/run/collect/audit，复用准确句柄、
设备lease、隔离本地存储和单次尝试生命周期。preflight单独挂6卡执行HF面板并退出，
随后双TP1服务挂6/7卡；client不挂NPU，只读取无标签输入和缓存，以macro128提交，
rollout32分配到两实例。服务batch32、token预算131072；除目标候选宽度及策略外，
原父服务的全部执行设置保持一致，包括精度/runtime/processor身份。

全量结果收集后独立重建完整macro日志及所有请求，再按照已绑定面板索引从全量
predictions提取实际输出，与同策略HF reference逐项比较logits、prob、threshold与
top1。只有完整transport、准确容器scope/终态、源策略证据及输出审核通过，才在
full-transfer-audit.json写runtime_provenance_verified=true。此流程不评分，不将
面板精确一致宣称为全人口后端一致。生产评分入口现已接入
`collect_context_full_transfer_scores.py`，重新审核所有角色后读取gold；详见
[三角色评分说明](context-full-transfer-scoring-v1.md)。真实正式源曲线/目标全量
运行仍待完成，不将代码接线或合成测试视为目标指标已产出。

固定模板为configs/portable/operations/prepare-context-full-transfer.json和
context-full-transfer-operations.json，分别读取FULL_TRANSFER_PREPARATION、
FULL_TRANSFER_OPERATIONS本地宏。准备配置包含source_admission（源曲线配置、
源发布文件hash、角色及policy ID）、两份portable输入spec、project、output、
remote_root、operations_root和tag。封包默认finite_inference_dispatch_allowed=false。
不是用户批准要求；真实源曲线未完成，当前没有可派发的目标策略包。

验证：源策略重建与原collector28项通过3.10秒；全量封包、命令隔离、HF实际输出
比较和源准入首轮28项通过3.18秒；增加完整服务设置/runtime校验后47项通过3.65秒。
测试执行真实封包、hash、配置重定位、源曲线重算、命令构造和有限输出比较；原模型、
实际容器和输入人口在部署测试中是替身，不能代替NPU及全量评分证据。

实际父包隔离导入检查保留1098个原Python文件逐字节一致，新增10个客户端/评测模块，
不修改正在运行的封包。首轮python -I屏蔽用户site导致缺torch，保留report.json；
这是本机依赖路径问题，非固定镜像/NPU故障。第二轮显式加入已安装第三方依赖路径，
结果记录operations/full-transfer-source-validation-v1/dependency-aware-import-v2.json。
第二轮52867终态0，客户端、parity和dual runtime导入成功。
该目录仅为导入验证副本，未封正式部署manifest，不可用于派发。
