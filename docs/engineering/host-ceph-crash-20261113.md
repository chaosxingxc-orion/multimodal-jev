# 175与181内核崩溃：同一Ceph同步路径

2026-11-13核验：两台机器均记录宿主机Ceph内核空指针异常，随后软锁死。
故障位置和当前模块指令高度匹配已知`req->r_session`竞态；没有证据把直接故障归为
模型算子或镜像内用户态库。没有vmcore，尚不能确定具体触发文件、容器或排除所有间接原因。

| 项目 | 175 | 181 |
|---|---|---|
| 故障时间 | 2026-11-13 09:36:27 +08:00（01:36:27 UTC） | 2026-11-12 16:00:26 UTC（13日00:00:26 +08:00） |
| 内核 | 5.15.0-25-generic #25-Ubuntu | 同左 |
| 异常 | NULL address 0x8，Oops 96000004 | 同左 |
| PC | unsafe_request_wait+0x1d0/0x564 [ceph] | 同左 |
| 系统调用栈 | fsync → ceph_fsync → unsafe_request_wait | 同左 |
| 当前线程 | python，PID/TID 137893，CPU122 | python，PID/TID 2147538，CPU170 |
| 后续 | Ceph注册请求自旋锁等待，持续soft lockup | Ceph注销请求自旋锁等待、soft lockup、网卡watchdog |

175最初拒绝SSH；随后重启恢复。普通账号无内核日志读取权限，经已授权Docker用固定
本地镜像、只读单文件挂载、无网络/无NPU、最小读取权限导出kern.log。
181持久journal直接保留故障栈。两台均未加载crash kernel；181未发现转储文件。

当前两台Ceph模块SHA256相同：
`42476bf3e27b084b1d495be72e2bf1721b7ab2d6ee70f7089d6fc092e012b955`。
加载模块与磁盘srcversion均为`B9055F931BF5E1EFBA6B57B`。
181反汇编再次确认故障指令前无空指针检查：

```text
unsafe_request_wait+0x1cc: ldr x0, [x26, #832]
unsafe_request_wait+0x1d0: ldr w2, [x0, #8]
```

上游[7acae6183cf3](https://github.com/torvalds/linux/commit/7acae6183cf37c48b8da48bbbdb78820fb3913f3)
明确修复请求进入unsafe列表后尚未赋值r_session的空指针；
[5bd76b8de5b7](https://github.com/torvalds/linux/commit/5bd76b8de5b74fa941a6eafee87728a0fe072267)
进一步用锁保护请求转发/重发时的会话变化。本轮重新获取两份原始patch并存hash。
omni在10月31日、11月1日已有同签名记录，本次模块hash和指令仍相同；仅重启没有修复缺陷。

容器共享宿主机内核，换镜像不能替换这段ceph.ko。应用并发写小文件/目录同步可触发
缺陷，但合法fsync本身不应使主机崩溃。两台驱动均26.0.rc1，181固件9.0.0.0.205；
尚未完成内核升级候选与Ascend支持矩阵认证。181 apt候选5.15.0.153.153只是仓库候选，
不能据此宣称兼容或直接安装。

## 并存问题与证据限制

181崩溃窗口rsyslog反复报ENOSPC；当前/var约99%、/data约100%，Docker约5184个
容器，容器JSON日志合计13,530,409,015字节，许多无轮转。日志仅占总用量一部分，
完整Docker分层大小扫描客户端超时，不能声称已定位全部空间归属。175当前/var48%、
/data67%，所以“磁盘满”不能解释两台相同的空指针指令。未删除容器、镜像、日志或checkpoint。

固定镜像CPU探针2368b744已核实01:35:57 UTC退出1、OOMKilled=false，早于175异常
约30秒；这不足以证明具体触发来源。记录失败终态，不重启。之前SSH不可达不等于探针一直运行。

## 已落实与剩余工作

- 已保留两台原始日志、模块身份、反汇编、上游patch及Docker库存，证据manifest见
  [结构化记录](host-ceph-crash-20261113.json)；私有原始日志不提交仓库。
- 已写[持久化规约](runtime-storage-policy.md)，保留本地fsync可靠性，拒绝旧主机Ceph热写。
  已对本项目instruction入口增加写入前检查，9项存储测试通过，连同instruction恢复/日程回归共61项通过；入口子进程验证在mkdir前拒绝不完整路径配置。不等于所有入口/工程已迁移。
- 175本地盘有容量，可作为后续暂存候选；181需先定向归档/清理、恢复预留空间。
  不自动全局prune或停正在运行的其他服务。共享模型/数据只读，本地热状态，归档由健康客户端负责。
- 永久修复需受支持内核升级、重启验证、kdump配置及隔离验收；未执行这些系统变更，
  未进行生产故障压测，也未启动训练。完整容量归因、业务触发线程映射和主机兼容矩阵仍待完成。
