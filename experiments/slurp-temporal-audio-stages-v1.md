# 当前 SLURP 策略的音频中间状态诊断

本实验承接 [概率诊断](slurp-temporal-rl-parity-v1.md)：同一当前策略的三个 HF 审计臂均未达到最大 logprob 误差 0.02 的门槛。目标是测量误差首先出现的位置；本次不训练，不读取 selection/TEST，不改变概率门槛。

固定使用上一探针按录音路径排序的第一条 learning 录音及其三个角色提示，未按正确性或误差挑选。直接音频、同一合并 checkpoint、BF16、物理 NPU 4、同一固定镜像；输入为 actor_features，采样器继续验证展开后 prompt token IDs 完全一致。完整参数和代码清单见[启动凭证](../docs/engineering/slurp-temporal-audio-stages-v1-launch.json)。

只记录第一次真实执行的卷积输入、三个卷积输出、conv_out、首层输入、18 层输出、ln_post、projected 及位置缓冲区。相同形状时比较逐元素最大/平均绝对误差、有限性和幅值；形状不同只报告形状，不通过截断或重排制造一致。HF/vLLM 的 padding、packed sequence 和算子路径可能不同，逐层误差只能定位后续实验方向，不能单独证明最终概率误差的因果来源。

修复旧探针在 callable 序列化之前退出的问题：钩子放入可导入的项目模块，传递本地固定 partial；仅本次无网络隔离容器设置 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`，不修改镜像默认值或生产采样器。模型、数据、代码只读；无外部 callable 输入。钩子不返回替代值，保存时 clone，避免后续原地修改污染快照。

宿主 torch 下的三个测试验证首调用快照、输出对象不被替换、后续原地修改与重复调用不改写快照、tuple 输出保持；3 passed。项目环境相关音频输入和采样器回归 12 passed。启动不等于采集完成或概率门槛通过。

运行目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/slurp-temporal-audio-stages-v1`。
容器：`94c16d85ffbba2941e78cf9b5f860971bf0a42ad340893a0c69a0f19bad3da25`。
整体超时 900 秒，已建立一个 `docker wait` 完成句柄；遵循 600 秒状态检查间隔，收到完成通知时读取结果。当前尚未核实采集结果。
