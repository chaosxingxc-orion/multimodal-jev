# Q7独立验收解除资源等待

2026-11-12 UTC。精确捕获的阻塞容器0d846415…使用与已审计完成监听器完全
相同的镜像、脚本SHA256、host PID、挂载及自动删除配置，未声明设备。
唯一差异是固定监听目录`/var/tmp/s1-bl-candidate-ops-media-v1/runtime`。
新版本仅允许该精确目录，不允许子路径、任意同前缀目录或不同脚本。
后续主机设备FD扫描保持必需，没有停止其他工程任务。

34项本地资源及接续测试通过，新增同脚本固定目录/错误子目录区分，以及
第二次接续从已superseded祖先寻找原训练派发凭证的测试。祖先配置、launch、
superseded必须被固定哈希，训练工作量一致；不重写旧凭证或重跑Q7训练。
该次资源修改未单独运行固定镜像测试；此前固定镜像证据不能替代本次测试。

在本项目4–7卡创建锁内，以完整argv与pidfd停止18个仅等待的控制器，
旧目录记录superseded而非成功。冻结新源码、配置及FSC来源引用后逐个启动，
18个新进程均确认完整argv存活、全部配置pins通过。任务顺序、训练数据、
奖励、学习预算、模型、训练/suite快照和FSC七策略矩阵不变。

Q/P现为intent-qp-d12-controller-v3，PID3688851。Q7独立suite容器
`30ab529762216b867fee9f512a29df6cc51fc35b0670326b498f779c6e76c171`
在12:31:31 UTC首次启动，12:33实际Docker状态running、镜像摘要匹配。
原Q7训练仍为653059e3…已成功退出的那一次，没有重复训练或重复验收。
只有suite完整退出并通过独立验收后才能推进Q8，不把启动视为学习结果。

关键下游：utility完整曲线控制器v3、resident suite控制器v3、FSC矩阵及报告v7、
真实行为消费控制器v2。所有新旧映射、PID、源哈希和实际容器状态见
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-resource-listener-migration-v1/`。
[接续验证凭证](intent-resource-listener-handoff-v1-verified.json)。
