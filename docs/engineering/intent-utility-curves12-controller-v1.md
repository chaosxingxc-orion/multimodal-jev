# 三臂完整曲线自动执行链

2026-11-12 UTC。为已冻结的correctness、smoothed-utility、uniform-credit各12步
协议增加自动执行入口，不按学习分数选择实验臂或检查点。

控制器实际PID3667057，等待`intent-resident-two-update-controller-v2`。
该前置链包含原三臂两步、常驻vLLM切换和两更新actor回放等工程验收。
等待使用600秒间隔，精确验证argv；失踪或失败不当成功。

## 派发前置条件

1. 精确前置常驻回放的真实训练/suite容器exit0及独立验收。
2. 三臂原两次训练与suite全部真实exit0、正确策略和更新游标。
3. 独立无设备CPU容器加载correctness第2步及原D2的实际FP32 master、Adam、
   BF16导出和候选概率，要求精确匹配；宿主观察实际exit0后才能继续。

CPU重放配置已冻结，输出：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-utility-correctness-replay-cpu-v1/`。
当前只验证了配置和代码，尚无该真实重放结果。

控制器之后顺序运行各臂第3–12步，每步使用原续训、完整独立gate与状态验收。
保留第0–12步全部指标和每步比较，输出各臂curve及整体curves.json。
某步失败直接终止，未知派发结果不自动重建。它仍是分步容器流程，不能称
常驻在线trainer，也不是独立泛化评测。

## FSC排期

只替换两个尚未派发的FSC等待进程：matrix v2→v3、external report v2→v3。
旧进程用精确pidfd停止，旧目录新增superseded而非成功terminal；配置/源码不改。
新matrix仍核验原lower-LR D12，再核验三臂全部36组训练/suite和correctness
重放实际成功，才执行原7策略×6视图×3793音频矩阵。不把utility策略加入
这版FSC评测，不根据曲线成绩改变原矩阵。

实际新PID：matrix3667064、external report3667069；三个调度器均核验存活。
35项本地协议/队列测试通过；固定镜像无设备容器15项测试通过，真实重放
配置的依赖均在容器内可见；实际curve配置也经执行入口完整检查。
[启动与验证证据](intent-utility-curves12-controller-v1-verified.json)。

控制器根：`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-utility-curves12-controller-v1/`，
保留准备和迁移脚本。当前等待不等于新增30步已经执行，收益仍待实际结果。
