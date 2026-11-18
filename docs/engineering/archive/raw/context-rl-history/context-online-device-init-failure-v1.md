# 连续RL双容器设备初始化失败与主线接续

同容器v2候选条件队列PID441409已启动，等待原probe独立collector426809完成。
只有已收集成功的物理7服务/物理4–7 actor映射证据通过、共同租约取得且设备空闲，
才派发新容器的两步工程运行；旧v1失败源码、产物和369pins不变。
新候选独立collector绑定372文件快照、监督器和服务终态，复用原算法审计后缀；
本地AST/prepare-only及缺probe拒绝检查通过，尚未实际运行/收集。
证据`context-online-shared-queue-v2.json`；当前无连续RL/恢复/正式Stage2准入。

新增未派发候选 `tools/jev/context_process_supervisor.py`：同容器内服务与actor由同一
监督器持有独立进程组，actor未连接就退出时使用原sequence0 close协议收尾；连接中
退出依靠EOF，无法正常退出才有界清理本次创建的进程树，不重试训练或接管已有任务。
actor训练时间不设截止，只限制服务启动和退出。8项真实CPU子进程/Unix socket测试
覆盖早退、正常退出、服务启动失败、服务中途失败、超时和陈旧ready拒绝；证据
`context-process-supervisor-cpu-v1.json`。未接入旧候选、未改变369原pins或364冻结文件，
不证明NPU共驻或连续RL成功。设备探针driver425576和collector426809仍等待原租约。

2026-11-14，固定镜像52ba94e1。actor a0198f4b于15:34:14 UTC exit1/nonOOM，
四rank均在`torch.npu.set_device`→`aclInit`报507899/Resource_Busy，尚未进入
模型加载、HCCL组初始化或RL更新。原始日志已由独立collector保留在
`context-online-npu-v1.json`及其collected目录；恢复重放未派发。

service b685080b此前已创建ready socket、独占容器可见物理7；actor容器可见4–7，
两个容器共用物理7。当前证据将问题定位于跨容器重叠设备初始化阶段，尚不能
从通用Resource_Busy证明驱动具体策略，更不能直接认定镜像损坏、OOM或模型问题。
同镜像单服务已启动；关闭服务后同镜像单DP4 Stage1四rank均通过set_device并进入load，
支持优先调查运行拓扑而非更换镜像。暂无宿主驱动根因认证或修复验收。

actor已明确终态，不能让专属service一直等未连接的actor占卡。先在宿主尝试
正常close因socket0600权限被拒绝，未改权限；随后在原service容器内发送
sequence0 close，收到ok，service正常exit0于15:35:56。证据
`.local/decision-objective-research/context-online-npu-v1-orphan-service-close.json`。
没有停止其他模型服务、复位设备或重启训练。原连续验收driver已停止并保留handles。

共享租约释放后，原Stage1接续自动创建04dd31b748d4，15:35:58 UTC启动。
remote1794670592.483905实查running/nonOOM，固定镜像、DP4物理4–7、global128/
micro8、step2→634、完整本地音频镜像/缓存路径与原配置一致。四rank进入load，
rank0已prepare_full_schedule；尚无完整训练结果。普通下次>=1794671192.483905。
原driver373142/选择373145继续，不重启或改冻结训练。

下一项RL修复限定于设备运行拓扑和失败清理：验证同容器内隔离进程能否服务逻辑卡3
与DP4共享，或使用平台明确支持的容器设备共享机制；先实际最小设备初始化检查，
再复用原两更新/恢复验收，不能把换拓扑或只启动成功称连续RL通过。新版本另存，
原v1失败与369文件pins保留。不得为验证抢占已运行Stage1或修改其配置。

## 同容器探针已排队

从固定镜像实际读取vllm_ascend/platform.py与device_id_mapping_demo.py，
该版本通过CANN `torch.ops._C_ascend.get_physical_device_id`查询实际物理卡，
并要求ASCEND_RT_VISIBLE_DEVICES在初始化前设置。源文件SHA和摘录保留于
`.local/decision-objective-research/context-device-mapping-source-v1.json`。
这是映射实现证据，未发现可直接证明本次跨容器失败根因的宿主驱动日志。

`tools/jev/probe_context_shared_devices.py`新增同容器最小探针：容器仅挂4–7，
服务进程先设ASCEND_RT_VISIBLE_DEVICES=3（候选容器内序号，必须实际读回物理7），
4个actor进程不设过滤，各检查运行时设备数4、物理卡4+rank，进行BF16小矩阵乘法。
服务保留上下文/分配直到四actor完成，然后再次计算检查。所有退出均回收本探针子进程。
尚无HCCL、模型、vLLM engine或实际共驻大内存验收，不能称RL修复完成。

driver425576已按同一设备租约排队，`context-shared-device-probe-npu-v1-status.json`
为waiting_shared_device_lease；启动前核对4–7空闲，不抢占Stage1。
脚本AST/CLI检查通过，冻结364/原候选369pins未变；准备证据
`context-device-sharing-probe-preparation-v1.json`。尚无远程探针容器或结果。
