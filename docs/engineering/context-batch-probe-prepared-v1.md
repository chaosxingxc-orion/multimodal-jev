# 批量推理有限NPU验收封包

本轮新增`context_batch_measurement_candidate.py`、`prepare_context_batch_probe_candidate.py`、
`run_context_batch_probe_candidate.py`，不修改正式训练及排队依赖。上一轮已完成服务和实际
CPU验证；本轮将候选推进到具有真实策略/输入、明确设备命令的有限验收包。

## 实验内容

使用固定Stage1策略9164eddffd9e、原生adapter和完整原processor，真实TRAIN首个global128
缓存输入hash 4b8791ee0abf。封包时重新验证joint policy及native adapter身份。
串行与batch4/token预算16384在两个新进程依次运行，保持BF16、相同定义/音频和原数值profile。
候选runtime独立标识，不宣称已替代原策略契约。

每种模式固定4轮：预热、两次计时、反序输入检查；输出恢复为原输入顺序。逐条验证身份、
概率与G16采样，重复及换序结果要求精确一致，串行/批量也要求精确一致，未放宽已有要求。
只测试当前完整128输入及固定策略，不包含更新/保存恢复/跨策略切换或任务效果。

通过实际worker的apply_model安装第一层语言attention与外层audio tower的pre-forward hooks，
记录实际语言forward次数、每次请求长度和声学调用长度；不把encode次数等同模型forward。
记录worker allocator peak allocated/reserved，不冒称设备总HBM。计时包含hook同步开销，
排除启动、策略切换、actor和checkpoint，若以后需要无hook性能需另报告。

比较器校验真实语言请求覆盖、完整四轮及相同输入/policy；报告概率/采样精确性、实际调用
是否减少和两次计时比率。调用减少与时间提升分开，不预先保证提速。

## 已有验证

8项测量本地tests通过/1.97秒：真实PyTorch hook、替身模型/计时、序列还原和比较拒绝；
3项封包/失败生命周期tests通过/1.91秒：完整树hash、额外文件/symlink拒绝、显式设备与
只读模型/源码挂载、serial失败保留日志/退出码且不执行batch阶段。
没有本轮真实Qwen/NPU结果，也没有固定镜像执行这11项的新证据。

实际封包位于`.local/decision-objective-research/context-batch-probe-npu-v1-prepared`，
1090文件，FILES.json SHA256：
`6d228b2c573d2e4c25e6a95952281002ed094327efb7270bbf29fe8cd3ef6564`。
独立重新读取完整树校验通过，包不含pycache。PREPARED.json给出完整Docker命令：
固定52ba94e1镜像、仅物理7卡、8CPU/64GB、网络禁用、Ceph模型只读、/data热写、/jev-tmp。
未创建远程目录或容器，未加自动队列；training_dispatch_allowed=false。
下一派发须核实固定镜像、设备空闲、独占租约与既有队列优先级，持久记录准确容器ID和终态。

## 资源及原任务

local1794718087两机npu-smi实际查询：175八卡仍被占用；181的5/6/7没有进程，4卡有
Omni同步进程551185。独立查询181固定52ba94e1镜像不存在，/data剩余110GB；当前仍有
Omni推理/同步容器。没有把“当时无进程”当作已取得整组资源，没有换镜像或抢占。
证据context-device-observation-1794718087.json与context-batch-npu181-resource-check-v1.json。

TRAIN原edf4509f live-v18 local1794718405.8137352 running/nonOOM，20992/40513；
下一普通>=1794719005.8137352。GRPO仍live-v39四rank240/634，下次>=1794718507.7995486。
七原本地driver实际ps存活，364基线及1078/1073/1167依赖hash无变化。
学术完整曲线、TEST和跨任务收益仍待，不以工程封包代替目标完成。


## 后续派发入口与镜像身份

`tools/reports/launch_context_batch_probe_candidate.py`默认只本地校验，实际1090文件包
再次验证通过。5项本地launcher测试已通过（见对应test_context_batch_probe_launcher_candidate.py），
不代表真实SSH/设备任务成功。入口要求既有MINDS队列完成、原设备租约、物理7卡空闲、
固定镜像和完整包身份，先持久化容器句柄再启动，观察失败不重建容器。
准确wait持有租约，失败产物保留。尚未派发、未增加自动队列。

181相同固定镜像已复制成功，并由原始配置SHA256与完整33层独立核验。
原严格Config对象比较失败是Docker inspect额外字段所致，旧失败记录保留。
见context-fixed-image-replication-181-v2-verified.json；此结果不证明181设备运行兼容性。
此前确认181 Omni B使用4–7卡，不能据一次无进程快照占用设备。

## 收集后的独立审核

新增`tools/reports/audit_context_batch_probe_candidate.py`为只读入口，不进入1090文件运行包。
接收prepared/collection/launch/output，先重验封包、准确容器/镜像、transport完整文件集与
原始archive hash、config、各进程退出码；再绑定真实128输入重算每条request身份和G16采样，
比较四轮串行/批量概率，核对顺序、实际语言前向长度覆盖、容量、有限计时及worker峰值。
独立重建时间比和前向次数，并与运行端comparison逐项核对；失败不产生成功收据。

11项本地检查通过/2.71秒，包括两侧同时记录错误采样/输入/长度、warmup遗漏、NaN时间、
内存记录异常、产物修改的拒绝，以及128条合成完整收集流程。CPU tensor/hash/采样为真实，
概率/telemetry/transport为合成，完整收集夹具的封包validate为替身，不是实际模型验收。
真实封包INPUTS清单另核对128条互异请求；原1073/1167依赖重新校验无变化。
证据：context-batch-probe-independent-audit-v1.json。未启动NPU、未改运行包或正式训练。

该审核只支持固定策略推理；前向长度覆盖不是设备kernel内部tensor trace，计时仍含hook开销。
即使通过，也不能替代策略切换、actor更新、恢复或端到端训练吞吐验收。
