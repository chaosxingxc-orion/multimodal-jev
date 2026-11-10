# 学术横向比较冻结方案 v1

日期：2026-11-09 America/Los_Angeles。关联 [学术报告](../docs/research/07-academic-baseline-lock.md) 与 [机器登记](../docs/research/academic-baseline-lock.json)。此文件锁定后续比较规则和必需对照；**不启动任何实验，不改动既有协议、运行状态、容器或 CPU 后台任务**。`locked` 指当前即可执行的研究约束，不表示某运行已经完成；`proposed` 指仍须生成具体配置、资源预算与启动凭证的实验。

## 1. 版本、假设与判定对象（locked）

L1 原生官方任务、L2 派生 schema 决策、L3 真实交互执行分别建表。每个结果键至少包括 dataset release、split、metric/scorer hash、输入模态、候选协议、训练来源、模型 revision、代码 revision、seed、解码配置、运行环境。键不一致不排名；文献报告和本地复测并列但不混算均值。

冻结以下可证伪假设：

- H1：类型化读出在同骨干/训练暴露下能提高结构有效性，但语义正确性需要独立验证。格式成功不算意图/槽位成功。
- H2：候选排列对齐训练能改善顺序稳定性，同时仍响应定义和语义改变。只有 flip/SKL 下降、正确率未保持时，不接受“决策更可靠”的联合结论。
- H3：音频直读相对 ASR→文本的差异来自证据利用。人工 transcript→同文本模型用于分离 ASR 错误和决策瓶颈，不预设人工文本一定上界。
- H4：RL 在固定预算下改善语义/执行收益。若只增加 rollout、训练 token 或设备时间，只能报额外成本收益，不能报同预算优势。
- H5：L1/L2 收益能转化为 L3 真实任务完成。没有完整模型驱动环境轨迹和官方终态评测，H5 保持未验证。

## 2. 必需基线与执行顺序

| 阶段 | locked 比较对象/交付 | proposed 执行与启动门槛 |
| --- | --- | --- |
| 0 数据与 scorer | FSC 官方三划分；SLURP real/synthetic 分离；SFC official+strict；MMAR publisher；退化/反例控制 | 只读 hash/ID/音频清单验证；所有 prediction 缺失、重复、未知 ID 的策略固定。Audio2Tool scorer 尚未达到官方复现门槛 |
| 1 FSC 官方基线 | SpeechBrain direct 官方权重复测；Lugosch frozen pretraining 重训参考；原生 joint intent | 先官方权重完整加载和 TRAIN 样例验证，再全 test 固定单次推理；重训先补预训练 checkpoint 与 recipe。当前主进程可能另行准备资产，此文件不操作它 |
| 2 SLURP 官方基线 | HuBERT direct 官方权重复测；原文 HerMiT 外部参考；real-only HuBERT 受控重训；ESPnet 2-pass 独立 intent 栏 | 必须分出 real+synth 与 real-only；检查 SpeechBrain 历史 recipe；完成 full 13078 预测 ID 集再调用原 scorer |
| 3 同骨干矩阵 | ASR→固定 Qwen3-0.6B、人工 transcript→同模型、冻结声学+head、0.6B 生成式 SFT、typed SFT、同预算 RL | full FSC / real-only SLURP 配置、任务接口和预算尚未冻结；不能直接把 31 录音诊断扩称全量 |
| 4 外部函数调用 | SFC test-ID/OOD 全级别、Audio2Tool 分 tier；共同保持 schema/多音频/历史 | SFC 原始协议为 FSC/SLURP-trained transfer。SpokenFC 内域 SFT/GRPO 重训单独 proposed，未授权以此文件改变原协议；Audio2Tool 补官方 scorer 或明确自实现适配 |
| 5 AudioJev | MMAR official checkpoint 原始顺序、Frozen Omni3B、MMAR 原生 Qwen2-Audio；论文三 seeds 独立文献栏 | 独立 CUDA 环境、模型可加载、完整 1000 输入、保存 logits/predictions/退出码；NPU port 需要新轨道与数值对照 |
| 6 扩展 | MAC-SLU；SLUE1/2 分任务；SpokenWOZ；MINDS/MMSU/MMAU/VoiceBench 分表 | gates、子集 IDs、隐藏评分和各任务依赖按登记补齐；不扩大下载清单 |
| 7 真正 Agentic | τ² voice-v2 + matched text-v1；FDB-v3 人声；EVA 后续 | API/provider/voices/judge 预算落实、模型驱动工具状态链验证；仅环境 oracle 不达标 |

阶段 5 可在独立 CUDA 资源落实后提前于阶段 3；表中顺序表示证据优先级，不要求占用设备等待。MAC-SLU/SLUE 等并非首轮必跑全量矩阵，不能以扩展集进度替代五个核心集的结论。

## 3. 数据隔离和输入契约（locked）

**FSC**：原生 train/valid/test=23132/3118/3793；以录音/说话人分组，衍生视图继承划分。官方 test 不参与 prompt、阈值、奖励、学习率或 checkpoint 选择。六视图分别报告；native joint intent 单独计分。

**SLURP**：real 50628/8690/13078；synthetic 69253 仅进入单列扩展训练条件。保留 scenario/action 原定义、实体类型和 filler，按录音计官方音频分母；文本 oracle 若使用 `--load-gold` 必须另列文本分母。重复文本/录音来源用于聚类统计。现有 transfer 域留出与全量原生任务独立，沿用现有 release hash。

**SFC**：固定 source `5cd9dc679584ba9b14454908a0f2c85e74c00147`。train/reference 不进入当前 Jev transfer 训练；test-ID/OOD 分表；全部 L1/L2/L3-1/L3-2 保留；两轮音频与 supplied history 不偷换成在线生成历史。官方 `NAN` 与本项目 JSON null、Noul 不是同一变量；按各自任务表示。confidence interval 按 conversation 聚类。新增内域重训需独立 train-derived validation，不能从 test 切开发集。

**Audio2Tool**：全是 test，不用于训练/校准/提示选择；ground-truth query 只给 oracle 条件。音频系统不见 gold call、reasoning、gold extracted params。保留 tier/源 query/音频变体映射，输出多调用的顺序和函数-参数绑定，先审计候选供给。现有文件数和卡片不同，以固定 manifest 为分母基础并报告覆盖。

**MMAR/MMSU/MMAU**：原始问题、选项及顺序、全音频不变；不能因空选项/非单选 reference/模型不支持长度而删除困难样本。特殊子集另报名单 hash 和排除理由，不以 short 子集代表 full。MMAU hidden 标签不入开发，先锁 checkpoint 再一次性提交；mini 也是 test，不是可自由调参的 dev。跨集以音频 SHA、来源视频/时间戳、query/文本去重记录审计；不得声称未做审计即无污染。

**L3**：τ 原 TRAIN oracle fixture 不用作泛化证据；gold criteria/用户私有目标仅 evaluator/simulator 可见。每个任务开始 reset 环境；调用通过真实 `get_response`，记录状态变化、错误、重试/重复副作用。固定历史回放不能替代新模型轨迹。FDB 自我纠正与未回应保留全分母；EVA 只跑 airline 时严禁套用三域结果。

## 4. 指标、退化控制与成功条件（locked）

| 层/任务 | 主指标 | 必需补充/反例 |
| --- | --- | --- |
| FSC native | joint intent accuracy | 三分量准确率、格式率、全分母、各 seed；多数类/固定类别 |
| SLURP native | official SLU-F1 | entity exact/Word/Char-F1、scenario/action/joint intent；完整预测集合 gate；不将 Word/Char-F1 简单平均 |
| SFC | 官方 Overall、function metric、slot F1，ID/OOD 分列 | strict call-multiset exact、function-bound slot F1、合法格式率、两轮 both-correct；数组改变/跨函数交换/重复调用/空输出反例 |
| Audio2Tool | tier-wise ordered tool accuracy、EM、micro slot F1 | normalized/strict 双列（若适配）、源 query 聚类；多调用顺序反例、gold 候选泄漏检查 |
| MCQ | publisher accuracy | exact-description 补充、覆盖、顺序 flip、对齐 SKL/TV；constant-first / oracle；不可满足 reference 保留 |
| L2 | 各视图 accuracy、paired both-correct | permutation 语义身份一致性；rebind 必须随定义改变；missing-candidate 拒识 P/R；Noul 相反问题 both-correct；all-null/first-candidate、随机/多数控制 |
| L3 | 官方完整任务成功/pass@1 | DB/environment reward 与 NL assertions/沟通评分分列；错误执行/重复副作用/撤销更正、延迟、轮次、tool/API 成本；pass^k 与 pass@k 分开 |

不变性目标各自配区分性测试：排列保持候选含义时应保持答案；重绑定含义、改变语音命令或补齐缺参时应改变答案。不能仅靠常量/全 null 取得高一致性。概率输出还报告 NLL/Brier、固定分桶 ECE、风险—覆盖；没有可比概率的生成/API 方法不虚构概率，相关栏留空。

完整性硬门槛：官方预定 split 的有效预测覆盖 **100%**，重复/未知 ID **0**，结果/原始预测/scorer hash 一致，终态 exit code **0**。失败和超时仍计完整分母；若 official scorer 不接受缺失，不能用剩余样本分数当完整官方成绩。覆盖不足只报告 partial，不用于胜负结论。

统计方案建议见下一节，不以文献小数差异判定我们复现成功。首次权重复测报告和作者数字的差值并排查版本、顺序、采样率、dtype、parser；不存在本文件擅自规定的“接近就算复现成功”。

## 5. 公平预算和统计配置（proposed，运行前再冻结）

建议同骨干训练 seeds **17/29/43**，继承开发序列便于溯源；全量 task split 不因 seed 改变。分别注册样本暴露匹配与资源匹配两种试验。前者固定训练录音/视图机会，后者先在 TRAIN-only 上测 throughput，再给每臂相同设备时限、有效 token/audio frame/rollout 总量的明确上限。typed 的候选评分、生成式解码、RL rollout/parity/reference passes、预计算特征及导出开销都计入。不同硬件仅作各自成本表，不用 CUDA 文献 latency 与 NPU 实测作公平加速比。

**本次预算数值未分配**：训练设备小时、每臂 token 上限、API 金额、运行 deadline 均为 null/TBD；不能凭“可用四卡”无限启动。以未来 TRAIN-only pilot 估算，不碰 test，不把估算当已消耗成本。建议每个官方静态权重先一遍完整确定性复测；训练方法报告各 seed 和 mean/sample SD；provider 重复次数与预算一起冻结，EVA 论文 k=5 / 扰动 k=3 若减少次数，明确标变协议。

建议配对 bootstrap **2000 次、95% CI**：FSC 优先说话人；SLURP 保留说话人/语句层级；SFC/SpokenWOZ 按 conversation；Audio2Tool 按 source query；MCQ 按去重音源；L3 按任务，同一任务各 trial 配对。多主检验用 Holm 校正；未预注册的切片只作探索。少 seed 的不确定性不能只用样本 bootstrap 替代。

建议联合判定：主指标差值 CI 下界 >0 才称该指标提高；“稳定性提高且语义保持”需另冻结非劣阈值，当前**不预填容忍下降值**；在阈值未锁前只能报告 trade-off。最小有意义差异、目标覆盖、ECE bins 和风险阈值必须在独立 train-derived calibration 上确定，不能以 test 结果反推。每个改动写新 protocol/run ID。

## 6. 运行栈和启动门槛

| 路径 | 已有证据 | 仍需满足 |
| --- | --- | --- |
| 本项目 Qwen3-ASR SFT/RL | CPU loader/engine/部分优化与开发协议证据 | 175 NPU **4,5,6,7** 已授权；主进程已验证设备隔离及单卡 BF16 音频前向/反向/LoRA 更新。NPU vLLM 概率一致性、原生 verl、导出/重载、分布式及断点仍需验证；所有对照同一已验证栈 |
| SpeechBrain/ESPnet | 官方 recipe/权重元数据 | 依赖隔离、模型资源完整性、CPU 或明确设备运行可行性；NPU 不自动兼容历史 CUDA recipe；旧库版本尽可能单独环境 |
| SpokenFC | 数据/scorer 可用，现有两音频请求诊断 | 原 Qwen2.5-Omni + ms-swift/LLaMA-Factory/vLLM CUDA 栈；论文 FP16/N32 与脚本 BF16/N8 显式对齐；移植另表 |
| AudioJev | 官方 source/weight pin，完整 MMAR input release | 官方 backend 显式 CUDA，隔离环境要求见 compatibility 文档；当前没有已确认 CUDA 分配。NPU port 必须单列，不改官方复现名称 |
| τ voice-v2 / FDB / EVA | 固定代码与部分场景/录音，τ oracle 环境组件验证 | 实际 provider model snapshot、凭据、voices、网络、费用上限、judge 版本、全链路模型轨迹；NPU 无法替代这些服务 |

正式启动凭证必须包含：代码和配置 hash、数据/模型 release、完整启动命令、镜像与设备/核数/内存、绝对输出路径、依赖版本、预算与 deadline、预期条数、scorer hash、随机 seed、预测原文和时延计量边界。长任务脱离交互会话，具有日志、心跳、checkpoint 和带退出码终态文件；不修改既有 CPU 任务。

启动前独立检查：TRAIN 小样本可走完模型→保存预测→官方 scorer；gold 泄漏/反例控制通过；与同臂 CPU/reference 的数值误差界已冻结；候选序列/音频不截断；任务全分母已知。不能用环境 import 或 oracle 通过宣称正式评测成功。

## 7. 与既有协议的关系和待决策项

继续保留 [FSC schema](fsc-schema-evaluation.md)、[SLURP transfer](slurp-transfer-v1.md)、[SFC external](sfc-external-evaluation.md)、[AudioJev-MMAR](audiojev-mmar.md)、[τ 环境桥](tau2-environment-bridge.md)、[算法开发比较](algorithm-comparison-v1.md) 和 text v1/v2 的原始数据、配置与结论。本方案不回写其训练池、scorer 或运行记录。现有机会匹配矩阵不是计算量匹配；新 full-data 比较必须另立配置。

待主进程决定：全量各臂训练/推理预算；独立 CUDA 资源或明确 port 轨道；是否进行 SFC 内域重训；Audio2Tool 官方实现缺失时是否采用自实现；L3 先跑哪个 voice-v2 cell/provider 及费用上限。上述决定影响运行，不影响本次学术登记完成。
