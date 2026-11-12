# 成对奖励的真实语音单步验证准备

本轮未派发新 NPU 任务。主线仍是 Qwen3-ASR-0.6B、BF16 未合并 q/v LoRA，
FP32 master/AdamW；既有 seed17 全量评估和 seed29 轨迹保持原协议。

准备根目录：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-pair-audio-pilot-prepared-v2`。
完整src/tools/task快照、协议、pair0/pair05配置与文件哈希已发布；v1保留为
仅有训练入口的早期准备，不使用其快照派发。当前镜像与已验证CPU verl一致。

两分支从同一事实SFT initializer、fresh Adam开始，各一次更新、8个请求、
4个来源、4对正反规则，取固定seed17第0批。三Span均逐批4正4负。
pair0 是 β=0 的同实现对照，pair05 是 β=0.5 候选；辅助事实奖励固定为0。
这不是原aux0.2三方法实验的直接同目标对照，也不据此挑选β。

新入口分为：

- `train_span_pair_pilot.py`：审核固定协议、初始完整runtime、输入/音频/策略身份；
  同一模型直接音频一次输出三Span，以Final成对正确奖励更新。更新前对所有
  Span执行完整候选/联合动作parity，保存原始行为概率、配对、优势和优化器。
- `verify_span_pair_pilot.py`：独立重算条件优势、裁剪loss、前后奖励与准入概率；
  核查112个实际Adam状态及step=1、FP32状态、checkpoint身份，拒绝提前宣称READY。
- `run_span_pair_pilot.py`：每分支更新后启动新HF进程，再启动原vLLM完整验收；
  覆盖初始验收的全部请求（固定probe及相邻训练输入），逐项概率阈值不变。
  HF重载在实际训练8请求上必须与更新后概率精确相同；所有候选及联合动作
  gate通过后才发布该分支验收。两分支均通过才生成整体READY。

为了复用现有推理入口，重载配置保留原transfer gate的人口描述字段；它们仅
确定待验收输入集合。新增 `engineering_protocol`/`engineering_pair_weight`
标识实际训练方法；不能把这些gate配置中的历史method字段当作训练方法证据。
真实算法身份来自pair协议、update记录和checkpoint training_state。

本地54项测试通过，包括数学梯度、batch归一化、成对语义/来源审核、BF16实际
参数更新、旧策略/错输入/概率偏移时零更新、损坏配对/优化器/准入/READY拒绝，
以及三方法比较队列失败拒绝。CPU mock替换仅用于单元测试；生产更新入口调用
真实verl。其CPU loss另已在固定镜像15组核验通过。

尚未验证真实音频NPU更新、该控制器的完整执行、独立导出重载或候选方法收益。
dispatch前必须核查卡5可用，并创建包含镜像/快照/协议身份的独占容器launch回执。
不得抢占现有评估，也不自动续训此单步探针。四卡恢复、学术基线复现和跨数据集
收益仍是独立未完成工作。

## seed17自动比较

新增 `span-transfer-three-methods-s17-comparison-queue-v1`，PID3497791。
00:46 UTC确认进程存活，绑定原SFT/GRPO/conditional三个完整评估容器ID，
等待退出0后核对源码pins，再调用既有独立全链比较工具。它不重启训练或推理，
不跳过轨迹、输入、策略或匹配预算审计。目标结果为
`docs/engineering/span-transfer-three-methods-s17-a2-v1.json`。
队列等待/比较中不等于结果已完成。

## 控制器和部署预检补充

新增7项控制器执行测试通过：任一训练、HF、vLLM子进程失败、独立验收不通过、
快照变化或卡号不符时，不启动第二分支且不发布READY；成功路径验证完整顺序、
原输入覆盖及HF证据固定。测试不执行模型。

`prepare_span_pair_launch.py` 已生成独立 Docker create/start 方案，见
`span-pair-audio-pilot-v2-launch-plan.json`。先create获取真实container ID，再
发布launch回执，最后start该ID，避免控制器读取尚未写出的回执。该方案尚未
执行NPU启动。协议引用的工程文档按哈希复制到共享存储并单文件只读挂载；
不依赖175服务器有本地工作区副本。

实际CPU容器预检v1因只读父目录与子目录挂载创建冲突未启动代码，保持created
状态作为失败证据。纠正挂载后的v2容器
`74e8ab645cc375c860b2b277e6ce97631ce9b1a9985a73ac93402969df5f853f`
退出0、无OOM、无网络、无NPU设备。两分支协议和全部源码pins复核通过，
训练/独立验收/控制器三个模块实际导入成功，初始策略原224请求的完整候选及
联合概率证据重新核验通过，最大log-prob误差1.19e-7。
这没有重跑模型，也没有验证新的NPU训练或新checkpoint。结果、容器和文件
身份见 `span-pair-audio-preflight-cpu-v2-{result,verified}.json`。

00:56 UTC进度：SFT、GRPO全量评估容器仍运行，两者已完成initial-learning和
final-learning各2520请求，正在处理selection阶段；SFT initial-selection已
完成1723/2304。seed29两个分支均验收5次更新，正在验收version6。
卡5尚不可用，保持等待原任务完成，不抢占、不重复派发。
