# 资源检查误判的定位与接续

2026-11-12 UTC。04:55按既定10分钟间隔核验时，step2验收队列与D12控制器仍存活，未派发第三步。对被报告占用的确切容器`6767685f2b385bbfb8534f9b6974a3ace70578aab24bd1bd0bfdf54c24ff69f8`进一步核查：它是Omni目录的inotify完成监听器，固定脚本只观察GENERATION_COMPLETE/ERROR文件，未声明设备。主机root只读FD扫描同时返回busy=[]、denied=[]。旧策略把任意privileged容器视为占卡，导致等待。

新增`resource_observers.py`只识别已审计的镜像摘要、确切chroot/nsenter/Python argv、脚本SHA256、限定监听目录、host PID、挂载与自动清理配置。任何显式设备映射、未知/变更privileged程序仍被拒绝；随后仍必须通过主机实际设备FD检查。不能仅靠名称或“无设备声明”放行。26项资源及控制器相关测试通过，覆盖脚本/参数/路径/镜像/设备/挂载变化和真实FD占用。

先冻结替代控制器源码，再持有本项目4–7卡创建锁，核验step2 suite无创建记录且step3–12均未认领。以pidfd及完整argv精确停止两个仅等待的旧控制器，确认退出，记录superseded.json；未停止他人容器，未取消训练或验收。替代控制器为`intent-step2-acceptance-resource-queue-v2` PID3552950、`intent-finite-d12-pilot-controller-v3` PID3552952，均已核验启动身份。旧快照及全部训练/验收计划不修改。

第二步suite首次创建并启动成功，容器`728b193a420a6c4393e8c93bdfd1dbb3f53bdd1eeee8f20c32ff720e8f841b9a`，实际Docker状态running；启动回执已归档。设置docker wait接收该确切容器完成通知，控制器仍按600秒间隔检查。此时只是验收已启动，不代表通过，更不代表独立任务收益。D12继续沿用原12步日程及每步完整验收。
