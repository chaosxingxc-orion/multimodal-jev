# 将验收镜像复用到181

本轮按目标落实跨主机共享同一工程镜像，未重建镜像或更换版本。上一轮有限验收封包是
具体进展；本轮启动实际镜像复制，并核实了181设备空窗背后的现有任务归属。

源为175固定`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`，
24,397,019,596字节、33层；181已有其中20层。复制前181 Docker根/var/lib/docker
可用155,611,574,272字节，/data可用117,790,167,040字节。无清理现有镜像/容器。

通过本地脱离会话的Python owner连接两个SSH，以docker save准确镜像ID流式传给181
docker load；不生成大tar中转、不写Ceph、不改tag、不占用NPU。
owner847611、source SSH847613、target SSH847614均实际ps存活；完整命令、stderr、load日志
与终态位于`.local/decision-objective-research/context-fixed-image-replication-181-v1-*`。
owner有独占flock，终态前不重建传输。当前尚未有成功终态，不能说镜像已安装完成。

独立finalizer848302已启动，实际ps存活，等待原owner同一锁后只做验证：
要求原save/load成功，再从181重新inspect，对比源/目标Id、Architecture、Os、Size、RootFS、
Config；成功时写`context-fixed-image-replication-181-v1-verified.json`。
它不重试复制、不创建容器、不占设备。缺失验证收据不能当成功。
启动证据：`context-fixed-image-replication-181-v1-started.json`。

## 资源归属的纠正

此前npu-smi在181看到5–7卡没有进程，但后续只读docker inspect已看到
`saea-b-instruct-local-storage-v1-r3`容器c29bf0a0运行，明确挂载物理4–7卡，并运行Omni TP4。
先前无进程只是服务切换空窗，不代表已释放资源。未终止或修改Omni，不将检查间隔当作
预留资源。相同镜像复制有独立价值，但NPU probe继续等待实际可用设备/现有队列。
证据：`.local/decision-objective-research/context-npu181-ownership-readonly-v1.json`。

## 当前正式任务

GRPO普通live-v40 local1794718554.6273518：同e564df04 running/nonOOM，四rank246/634，
64/128/192已提交；下一普通>=1794719154.6273518。TRAIN仍live-v18的20992/40513，
下一普通>=1794719005.8137352。1078/1073/1167依赖再次hash无变化。
没有新训练/模型评测或NPU批量吞吐证据，完整学术目标仍进行中。
