# 批量模型服务候选：容量分组与原协议复用

接续[批量owner](context-batched-owner-candidate-v1.md)，本轮新增
`tools/reports/context_batched_service_candidate.py`与`serve_context_batched_candidate.py`。
当前正式GRPO/RLOO、actor、任务、reward、配置及后续队列不变。上一轮实现及固定镜像
失败定位属于具体进展，本轮完成可供现有bulk客户端调用的独立服务入口。

## 行为

先验证整条命令（最多128个CPU输入），再按完整请求身份在本次命令内去重，按token长度
稳定排序，受max_batch_queries和max_batch_tokens双预算分组调用owner.score_many。
按原输入顺序恢复每条结果，包括重复行；保留全部训练曝光和后续逐行采样，不跨命令、
跨策略缓存概率。输入超过单请求容量时拒绝，不截断或静默退回原单条路径。

复用原BulkContextEngineClient、socket消息、原子payload写入及策略切换接口。服务单owner
串行处理命令，每个score_many仍使用owner的策略切换锁。所有批次完成且逐条身份复核后
才返回整条命令；失败保留payload并结束该服务连接，客户端failed，禁止重试继续。
重复结果独立复制，避免调用方修改某一返回行影响另一行。

新入口在模型分配前验证协议、显式candidate_runtime、存储、socket和依赖hash；发布ready
记录容量及worker安装收据，关闭后保存最后一批encode调用计数并清理socket/engine。
encode计数不等于实际scheduler模型forward次数，真实NPU验收须另测。

## 验证证据

本地28项通过/2.69秒，包括真实Unix socket发送、现有客户端切换策略、整批输入校验、
长度分组/去重/恢复顺序、G16采样一致、错误不返回部分结果、启动前拒绝和正常资源清理。
模型engine/策略加载及存储准入使用替身，不是实际模型启动。

相同28项在固定镜像52ba94e1的独立CPU容器ec6b264a执行，全部通过/27.69秒，准确wait=0，
独立inspect exit0/nonOOM。380文件payload全部hash重验；无NPU、无网络、非privileged，
本地/data热写/jev-tmp。完整stdout/stderr和身份收据见
`context-batched-service-fixed-image-cpu-v1.json`。不覆盖AscendMetadata实际导入或NPU运行，
上轮无NPU硬件查询失败保留原证据，未通过替身宣称已经解决真实后端准入。

另以既有真实TRAIN首个global128输入执行分组，重验inputs.pt hash及128条输入tensor身份，
不读标签/概率、不重新processor或模型推理：

| 候选容量 | 展开token预算 | 规划encode次数 | 每批实际最大token |
| --- | ---: | ---: | ---: |
| 4条 | 16384 | 32 | 11589 |
| 8条 | 32768 | 16 | 23071 |
| 16条 | 65536 | 8 | 45984 |

128条均唯一且完整覆盖，原顺序可恢复。见`context-batched-service-real-input-plan-v1.json`。
这些是输入分组计划，不能声称NPU显存可容纳、实际forward次数减少或已有加速倍数。

## 下一具体步骤

独立有限NPU封包/运行：同一固定策略及真实输入，串行对批量，检查完整概率与采样、顺序/
邻居不变性、实际worker分组、峰值内存和耗时。固定镜像CPU真实接口导入会触发设备查询，
需要有设备的运行验收。必须沿原资源队列，不抢正式训练或增加其准入门槛；尚未派发NPU。
当前工具没有自动加入训练链，也没有把候选profile冒称正式概率协议。

## 当前原任务

GRPO live-v39：local1794717907.7995486，同e564df04 running/nonOOM，四rank240/634，
64/128/192已提交；下一普通>=1794718507.7995486。
TRAIN live-v17：local1794717784.4613173，同edf4509f running/nonOOM，19968/40513，
下一普通>=1794718384.4613173。七个原本地driver实际ps存活；1078/1073/1167依赖无变化。
完整学术RL曲线、TEST与跨任务迁移仍待，不以候选工程测试代替研究目标完成。
