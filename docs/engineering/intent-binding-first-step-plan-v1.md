# 重绑定首步训练计划与实际容器预检查

2026-11-12 UTC。已在共享存储冻结首步计划：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/intent-binding-first-step-v1`。
该目录包含protocol、training与suite的config-template、源码快照、精确Docker
创建命令和挂载哈希审计。没有生成正式训练config，没有创建NPU容器或新队列。

首步固定D臂、seed317、lr1e-4、原始temporal adapter与新建Adam，8个录音的
4种候选视图共32请求。来源顺序和每来源总权重与原D首步相同，Final奖励不变。
这单独检验候选绑定增强的工程可运行性，不改变已有D/Q/P曲线，不作为FSC选择
模型，也不把4视图额外计算误称等算力。完整连续收益实验仍须独立协议。

训练使用175物理NPU4–7、FSDP FULL_SHARD、BF16 LoRA执行/导出与FP32
master/Adam；更新后suite使用物理NPU4、vLLM TP1及完整712输入HF/vLLM核验。
固定镜像与现有训练相同。命令内训练预算3600秒、suite预算14400秒；它们是
进程执行上限，不是预估完成时长，观察超时不能触发自动替代任务。

两个CPU容器实际复用了各计划的文件挂载和环境变量，但移除所有设备、驱动
挂载与模型执行。它们检查全部固定输入/源码的哈希、真实学习来源日程、训练
及验收模块导入和正确入口命令，均实际退出0、无NPU挂载：

- training：`c13136906ef1134914e77aa2d6106c69f10ba233cd664a2a74613467cd987d76`
- suite：`03e9f74c019edf21a12529c7ec8945b9d655c36975d1f83d510915f1cb25818e`

本机/home审核证据已逐文件复制并映射到相同容器路径，不依赖175上的同名目录。
这些检查证明冻结计划的挂载、导入与数据可用，不能证明NPU显存、模型前向、
真实策略验收或训练效果。见`intent-binding-first-step-preflight-cpu-v1-verified.json`。

新增`freeze_intent_binding_training.py`只在真实712输入gate退出0且全量认证通过
后，固定其launch、两侧概率、阶段终态及READY，才发布正式config。缺证据、
178人口、挂载失败或已存在创建尝试时拒绝；19项CPU冻结/观察测试通过。
该工具尚未对实际计划执行，已冻结源码快照保持不变。

下一步实现并验证专用队列控制器，排在当前beta1zero连续控制之后。派发必须
重新核验实际前驱完成、原step0绑定gate、完整计划哈希及物理卡所有权。首步
真实训练和更新后独立验收通过前，不推进更长重绑定训练。
