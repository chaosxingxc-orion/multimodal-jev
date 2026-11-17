# 当前概率策略切换候选与缓存任务修正

2026-11-14。保持context-prob-v1冻结的356个训练源码和两份配置不变，
`verify_context_baseline_lock.py`再次验证364份绑定文件/证据/归档全部匹配。
以下为独立Stage2候选，不属于已发布基线，不改变现有排队Stage1。

## 头部切换组件

`tools/jev/context_policy_activation.py`提供CPU资产准备与头部两阶段切换。
prepare_switch校验完整joint snapshot/native adapter，要求contract、冻结marker和
processor资产完全一致。HeadActivation先检查当前头/marker与策略一致，再关闭
readout请求身份、装载并回读新头。只有匹配的worker adapter回执才能提交新身份。
任一异常使会话失效，不能重试或继续服务。A→B→A的完整头权重恢复测试通过。

`tests/test_context_policy_activation.py`共8项CPU测试通过，包含旧/新请求在stage
期间均拒绝、错误当前身份/形状/NaN/marker/contract、错误adapter回执和反向切换。
这不是实际vLLM/LoRA事务：调用者仍须串行化与排空请求、验证worker原生adapter
确实加载、更新模型所属policy身份并执行真实音频概率验收。回执不能来自用户自报ID。
prepare_switch文件认证复用已有模块；本轮8测试主要验证头状态机，不能泛称全部
资产准备与实际worker原生LoRA切换已经独立验收。连续RL尚未派发。

## 输入缓存工作

13:27:07 UTC到期观察：原9822c31d running/nonOOM，3328/40513、1156.97秒。
同批观察确认辅助v1 5fa4bed4 exit1/nonOOM，启动时缺TRITON_CACHE_DIR，未处理输入。
第一次修正v2 a5d68bb4明确exit1：/tmp为overlay，不符合本项目本地存储约束。
两次失败完整证据分别context-learning-cache-assist-v1-failure.json和v2-failure.json。
这些是辅助launcher配置错误，不能称为镜像或原训练失败；未停止/重启原writer。

v3 b4b40b0a已于175远程13:28:31 UTC启动，exactwait session17861。
启动前在同一mount/env命名空间执行require_training_storage，所有热路径和HOME日志
均核实位于/data的可写ext4挂载；原source/mirror只读、共享cache单独读写。
脚本launch_context_cache_assist_v3.py保留created/launch身份，避免重启未知状态容器。
本轮尚无v3完整进度或终态证据，不能声称输入处理/加速成功；下一普通观察>=600秒。

原输入driver281383、全量Stage1 driver307877、选择集driver310979继续，
完整40513缓存的发布与独立检查仍由原流程负责，辅助writer清单不替代它。

## 原生vLLM接入与实际验收启动

固定镜像源码核对：vLLM的worker.add_lora→model_runner→LRU manager负责加载、
驱逐旧adapter及激活单槽；原生pack将scaling乘入B并把q/k/v打包。
源码摘录.local/decision-objective-research/context-vllm-switch-source.txt。

新增独立工具context_switch_worker.py与context_switch_rollout.py：
owner串行化score/switch；worker先关闭readout请求身份，加载完整新head，
调用原生add_lora，然后核对已注册/活动slot、q/v完整112矩阵、B缩放及缺失k；
最后提交head身份和模型policy ID。任何异常使worker/owner失效。
检查覆盖已加载CPU LoRA张量与活动槽，尚不等于验证所有NPU buffer内容；
真实音频概率对比单列，不能只用这些receipt声称完整策略已生效。

probe_context_switch_npu.py使用已独立收集的admitted step1/step2，
先在不同进程生成各自HF16音频参考，再一个vLLM引擎A→B→A，
比较每次全部60概率、输入身份，并要求A/B至少一个音频输出有区别。
测试不训练、不消费gold、不将小样本认作任务收益。

实际容器d368109e已在175物理7卡启动，准确wait session40628持统一设备锁；
独立collector驱动PID323640，状态context-resident-switch-v1-status.json。
collect_context_resident_switch_npu_v1.py会核验策略张量、adapter、48条输出与HF概率；
失败时保留完整终态日志。当前尚未终态，不重复派发。
20项既有头部状态机/在线组批CPU回归通过，364份冻结基线文件仍匹配。
真实vLLM切换、完整连续actor更新恢复与RL收益仍待最终证据。

首轮d368109e已明确exit1/nonOOM并独立收集：A/B HF阶段均exit0，
vLLM owner初始化前因独立工具仍引用相对包路径而ImportError；未执行策略切换。
证据context-resident-switch-npu-v1.json。已修为绝对包导入，只改独立工具。
v2 f6dc19cb物理7卡running、exactwait96590持共享租约，collector driver324510，
状态context-resident-switch-v2-status.json。v2复用首轮成功的A/B HF与策略文件，
逐文件hash绑定后只重跑真实切换，避免重复音频HF计算；尚未有终态。

13:37:51 UTC普通到期观察：原cache9822c31d与assist-v3 b4b40b0a均running/nonOOM。
原writer5120/40513、1791.19秒；逆序辅助4608/40513、494.42秒，均0hits。
辅助已实际处理输入，约9.32条/秒，原累计约2.86条/秒；处理样本、起始时间和
共享负载不同，不能直接称3.26倍因果提速或外推训练吞吐。最终全量缓存仍待原发布与独立收集。
下一普通远程观察不早于上述观察+600秒。


终态补充：v2 f6dc19cb exit0且独立collect通过，详见context-reference-cache-v1.md与
context-resident-switch-npu-v2.json。16音频A/B/A共48向量HF概率0差、两次切换约.795秒；
旧“尚未终态”条目为历史，不再等待/重启该handle。连续actor闭环仍待。
