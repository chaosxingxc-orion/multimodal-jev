# 两步真实语音常驻 actor 回放准备

2026-11-12。已验收的D第1/2步当前通过独立容器和FP32 master/Adam恢复执行。
新增回放保持同一FSDP模型、原生verl actor与Adam对象常驻，执行这两步，
逐项比较历史参考；不改变来源、奖励、学习率、随机种子、BF16执行和FP32状态。

第1步复用原训练入口，新增默认关闭的after_update回调，在模型和进程组销毁前
进入第2步。第1步master/Adam、before/after概率和BF16导出必须与参考完全一致，
并认证原独立gate。第2步使用原已验收vLLM概率与原来源日程；计算更新前完整
候选概率，要求与参考一致，之后比较所有rank的FP32 master、Adam和BF16导出。
会话使用新增跨rank更新成功确认，最终保持awaiting_acceptance，不能自动接受。

这是actor常驻回放，尚未把在线vLLM切换接入更新循环。历史行为概率复用只因
策略/输入/状态精确匹配而成立，不是用actor重算替代rollout。后续需独立实际
HF/vLLM gate，再与热切换/设备驻留验证衔接。不能把本准备称为完整连续trainer。

准备根：/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-resident-two-update-prepared-v1。
冻结原两步训练和suite实际容器退出0、完整accepted证据、源码及本地证据挂载。
固定镜像无设备CPU容器实际退出0，34项测试通过，真实完整配置与挂载校验通过。
测试包括BF16导出相同但FP32 master一个ULP漂移、Adam漂移、错误rank/游标、
缺失状态和更换学习因素。详见[验证记录](../../../intent-resident-two-update-cpu-v1-verified.json)。

尚未派发NPU：需先完善独立最终gate与队列验收，保持原冻结队列顺序。
已有运行源码快照未修改，当前单步入口默认行为不变。

后续补齐独立验收并排队：新suite使用回放的resident-adapter，实际重新运行178输入
HF/vLLM，然后独立加载两步所有rank的master/Adam，比较原参考；并要求两个后端
的完整候选概率与同策略历史参考完全相同。训练内部passed不是独立验收。
宿主控制器必须观察训练、验收和历史参考实际容器退出0；任何失败均不重建或跳过。

固定镜像CPU验收测试22项通过，包含真实张量漂移负对照和mock gate失败路径；
所有真实计划pin在容器内可见。[CPU证据](../../../intent-resident-acceptance-cpu-v1-verified.json)。
这不代表真实NPU回放或gate已经通过。

控制器已启动，等待常驻vLLM A/B/A切换探针成功，之后派发DP4常驻回放及TP1独立suite，
600秒轮询。[启动记录](../../../intent-resident-two-update-controller-v1-launch.json)保存准确进程与源码身份。
原冻结队列不变，当前工作只验证已有D1/D2的执行等价性，不新增学习结果或SFT对照。
