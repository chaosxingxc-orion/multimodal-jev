# 后继任务15分钟监听

新增独立只读入口`python -m tools.reports.watch_context_successor_jobs`，后台PID954637。
原GRPO/inputs watcher PID913576继续负责当前两个句柄，两个监听器的目标集合不相交。
主实验、队列owner、启动器、设备租约、结果收集及选模逻辑没有修改。

## 固定范围

共26个预定目标：RLOO arithmetic-v2训练；GRPO和RLOO各11个完整selection评测点；
FSC pair、clean SLURP TEST pair、MINDS pair。标签严格来自已核对的现有driver命名，
不根据目录通配符自动收录其他任务，不会触发任务派发。

每60秒最多只扫描本地固定launch记录。新记录出现后绑定文件SHA、完整container ID和镜像、
remote_root，至少等待900秒才第一次远程读取；之后每次尝试至少间隔900秒。
远程超时和身份失败同样消耗观察间隔。首轮可能比容器启动晚15–16分钟；短任务可能在
第一次观察时已经终态，由原收集器正常收集，不因此补跑。

读取docker inspect后，只读取训练rank计数/耗时/commit名称，或评测的progress.json。
没有progress时返回空记录，表示尚无已发布计数，不推断为停滞。评测不读取预测、gold
或结果分数，不能用TEST进行配置选择。已创建但未start的容器不作终态。
只有成功核对原句柄并观测exited/dead才移出监听；失败不重建、启动或接管容器。

单实例flock，attempt在网络I/O前持久化。各目标首次发现、下一到期时间和终态持久化，
监听器恢复后继续沿原记录节流，原launch内容发生变化则拒绝跟随。
所有既定任务观测终态后自动退出；未启动或前继失败的任务保持等待，不擅自启动。

## 当前实测

22项本地测试通过、无跳过，包括原observer/watcher回归及13项后继检查：900秒节流、
缺launch无remote、容器created不误判终态、运行/终态、超时/身份失败、launch替换拒绝，
实际Python进度读取脚本和排除分数文件。远程传输为替身；尚未出现真实后继句柄，
不称已完成真实后继远程周期。

启动后通过/proc核对954637及原913576实际存活，首轮26目标全部awaiting_original_launch，
零remote attempt。证明见`context-successor-observer-v1.json`。当前GRPO最近仍为331/634，
下一普通观察1794727536.5727654；inputs最近34048/40513，下一1794727354.8090281。
本次没有提前查询远程主任务。

状态：`.local/decision-objective-research/successor-observations-v1/watcher-status.json`；
日志同目录`watcher.log`；逐目标`*-state.json`、`*-attempt.json`、`*-result.json`。
此进程只采集，不自动产生科研判断，也不修改简单收益/复杂训练准入门槛。

364项原锁及1078/6/2/1097/1073/1167六组运行和排队依赖复核无变化。
