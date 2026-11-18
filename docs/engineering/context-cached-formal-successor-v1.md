# 四卡缓存 GRPO 正式训练后继

本入口只执行简单SLURP全量正式训练，不处理复杂任务。算法、数据、模型、精度与
LoRA配置沿用已准入的两阶段基座；控制器不根据selection成绩改loss、阈值或初始化。

## 固定顺序

1. 等待准确的Stage1完整selection驱动退出，检查driver/config SHA256和成功终态。
2. 重新独立审核全部11个selection点及发布曲线，保留最优点与固定318终点的区别。
3. 重新审核同一runtime的100步效率证据，包括其有限验收和固定reference来源。
4. 复用`context_cached_formal`封包、launch、collect及独立审核，生成新的正式运行包。
5. 全量634更新完成后，只有81,026真实曝光及完整更新产物审核通过，才记录训练成功。
   Stage2完整selection、TEST和FSC/MINDS迁移仍是独立待完成事项。

Stage2从Stage1的318终点重新开始，reference是该终点的全量TRAIN固定输出缓存。
不会从100步效率终点续训，也不会将Stage1 selection最优点换成reference。
正式训练为两轮40,513条clean TRAIN，DP2/global128/micro8，两个TP1推理实例service32，
rollout32、macro128；BF16执行，FP32 master/Adam/loss。所有输入仍直接来自音频及任务定义。
预登记点为0、64、128、192、256、320、384、448、512、576、634。
每步发布可训练策略供推理刷新；optimizer仅在预登记非零检查点保存，主模型只以身份引用。

## 配置与运作

入口`tools.reports.context_cached_formal_successor`，模板
`configs/portable/operations/context-cached-formal-successor.json`读取
`${CACHED_FORMAL_SUCCESSOR_CONFIG}`。字段为：

- `predecessor`：Stage1曲线config、driver、terminal准确路径及前两项SHA256。
- `experiment`：新package、remote_root、operations_root及唯一tag。
- `output`：本地后继记录目录。
- `formal_training_dispatch_allowed=true`：显式正式训练派发开关。

共享host/lease/存储位置及release_evidence从已审核效率部署继承，保持四卡租约。
新路径不得覆盖曲线、效率、有限验收、reference或父runtime目录。配置内机器绑定集中在
不入库的本地宏；共享模板不含个人地址。使用准确pidfd等待，不轮询前驱或调用模型调度。
失败即保存结构化终态并停止，不重试、不改写运行包；重复启动拒绝覆盖已有记录。

正式训练调用期间复用`observe_context_replica_training`：每900秒最多一次准确容器观察，
初次容器不足900秒时不访问远端。观察失败仅记录，不改变训练；owner结束后停止监听。
监听不触发模型分析，结果统计/独立审核继续由固定程序负责。

## 验证与当前边界

47项本地CPU测试通过，包括正式封包/实际生成runner、稀疏optimizer保存、固定曲线、
前驱身份与篡改拒绝、selection最优点不替换318终点、部署依赖继承、错误人口/不完整更新拒绝、
失败停止和900秒监听。测试中的NPU、远端进程和完整曲线审核使用明确替身；
这不是新正式运行的NPU准入或收益结果。

实际本地宏为`.local/config/context-cached-formal-successor-v0.0.41.json`，配置校验通过。
前驱为`context-infonce-selection-curve-v0.0.40`，后继记录根为
`/srv/multimodal-jev-xingchao/operations/context-cached-formal-successor-v0.0.41/`。
正式目标`context-cached-formal-v0.0.rc6`仅在前驱真实成功并重新审核后生成和派发；
当前登记这个名称不代表rc6已发布、已训练或已通过NPU验收。

后继驱动PID2322614，启动记录1794952796.6019766；启动后现场ps确认存活，
等待准确的Stage1曲线前驱。终态为上述后继根下terminal.json，
正式训练准确句柄和独立审核结果写入其独立operations_root。

正式训练后的[完整GRPO源评测](context-cached-grpo-evaluation-v1.md)由独立固定后继负责，
继续等待本驱动准确终态，再重审训练和所有预登记策略。当前训练队列及配置不原位修改。
