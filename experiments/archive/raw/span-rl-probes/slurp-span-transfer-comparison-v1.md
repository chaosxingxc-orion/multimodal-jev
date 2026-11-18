# 三方法12步对照与完整迁移评测

沿用已固定 `span-rule-transfer-protocol-v1.json`，同初始事实SFT、seed17、
structured、aux0.2、12个相同均衡batch、lr1e-4、fresh FP32 Adam及BF16
执行/导出。每版本独立HF重载和vLLM全动作验收通过才进入下一步。

| 方法 | 运行 | 设备 | 当前证据 |
| --- | --- | --- | --- |
| 条件边缘PPO | span-transfer-conditional-s17-a2-v3 | NPU5 | 12步独立审计通过 |
| 候选交叉熵SFT | span-transfer-sft-s17-a2-v1 | NPU5 | 已派发，未完成审计 |
| 联合采样GRPO | span-transfer-grpo-s17-a2-v1 | NPU6 | 已派发，未完成审计 |

NPU4另行运行初始与条件方法末次策略的完整评测。其余任务有独立容器、
不可变源码快照和输出目录；只开放授权物理4–7卡。设备参数化保留TP1、
相同运行时数学profile与验收阈值。这里是实验并行，不能称作四卡FSDP训练。

同请求和更新数不保证等计算预算：SFT与条件方法逐请求累积梯度，联合GRPO
一次处理8个完整请求组、每请求32个采样动作；条件方法对有限候选精确积分。
报告各自总设备时间、token/音频暴露与验收开销，不把同时启动当作资源匹配。
不同方法的损失数值没有可比大小，不根据loss接近0判断梯度或训练是否有效。

评分分别覆盖learning/reserved来源与trained/heldout规则，共四单元。入口
校验四个单元互斥且穷尽、来源角色、实际规则真值表分组、请求/来源分母，
不能仅有相同预测ID就接受错分的task cell。初始与末次必须同模型基座、
同study hash，完整输入身份相同，末次固定step12，不挑选中间checkpoint。

预定主指标逐规则宏BA补充来源级配对bootstrap 2,000次；每次重采样保留
同来源所有规则。若某重采样缺少任一预定规则的某类gold，该次完整宏BA
未定义：明确报告次数，不在较小规则子集上算宏平均。有效次数不足100不
输出区间；其余区间标为以完整标签覆盖为条件的描述性区间。准确率保留
请求加权与来源宏平均两种差值区间。均不替代训练seed不确定性，也未进行
多比较校正，不据单个区间宣布算法优胜。

当前仍只有条件方法的训练链完成验收，完整迁移分数及SFT/GRPO比较待运行
完成。后续需扩展其余seed与单问/多问，并开展跨数据集原生任务；这一轮
SLURP内组合迁移不能替代原生SLURP学术复现和跨数据集泛化目标。

三方法统一比较入口为 `tools/jev/compare_span_transfer_methods.py`。每个方法先
完整核验训练链和初始/末次全量评测，再逐步比较实际请求顺序、seed、arm、
辅助权重、学习率、采样数和已记录token/音频暴露；只比较总步数不足以通过。
三方法必须有完全相同的初始预测及跨策略输入身份。输出全部四单元×三种
输入形式的表，并同时报告GRPO对SFT、条件方法对SFT、条件方法对GRPO的
配对差值，保留负结果，不只展示最有利的切片。预算核查13个测试包括错序、
中途fresh optimizer、不同起点、不同权重、数据成本和方法标签篡改。

若训练时延无独立单调时钟计量，只能从初始gate配置文件mtime到terminal
作带标签的近似墙钟统计，不能作为可信的纯训练速度或计算量匹配证据。

可复用启动器 `tools/jev/launch_span_transfer_evaluation.py` 先重算完整训练链，
再从study固定初始与末次adapter、全量learning/selection请求和源码快照；
`--launch`启动隔离的HF评测容器，否则只生成可审查的启动配置。四阶段由
`run_span_transfer_evaluation.py`顺序执行，各阶段推理后必须独立核验。

SFT与GRPO已分别挂接 `span-transfer-sft-s17-a2-evaluation-queue-v1` 和
`span-transfer-grpo-s17-a2-evaluation-queue-v1`。队列等待当前确切container ID，
不重启训练；退出0后校验已固定的本地源码，再调用完整审计与评测启动器，
在训练原卡5/6上运行。源码变化、等待失败、训练非零退出或审计失败均终止
队列并保留终态，不擅自另开训练。`dispatched.json`只表示评测已启动，不能
当作评测完成。后续改动src/tools前须注意已有队列的源码pin，不能静默改变
其执行内容。控制器/队列10项测试通过，实际全量评测结果仍待完成。
