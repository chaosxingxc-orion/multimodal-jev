# 全量 Stage1 与选择集评测接续

2026-11-14：上一轮已独立收集 admitted 两步训练。本轮保持已验收的356个训练源码
文件哈希不变，补齐实际作业之间的接续；不宣称全量训练、评测或RL收益完成。

## 已完成的输入工作

本地学习音频镜像 `b4d97c6d` 在175远程13:06:31 UTC exit0，
40513条/40513唯一blob全部重新hash并与固定学习人口对齐。
证据 `context-learning-audio-mirror-v1.json`，manifest
`e7069e6069162ea302d61d85183ab913777182c604e4b1c70dcf370504d29b02`。
原输入driver281383已自动启动CPU全量特征缓存容器
`9822c31d477612b92f960bf156a7b1794e6bcb59d55991daef7dc1c6ed08be7e`，
launch记录1794661688.47；不重复派发。全量缓存尚未取得独立终态证据。

## 接续驱动

全部脚本位于 `.local/decision-objective-research/`，运行日志及状态文件同目录。

- `continue_context_admitted_full_v1.py`，PID307877：准确等待上述缓存容器，
  成功后独立收集并验证40513条缓存；调用既有launcher的`--full`，
  从step2恢复到634；成功后调用`collect_context_admitted_full_npu_v1.py`。
  状态 `context-admitted-full-chain-v1-status.json`。
- `continue_context_stage1_selection_v1.py`，PID310979：等待前述进程锁释放并验证
  成功状态，顺序评测step634和step0；任一上游失败均保留原handles并停止派发。
  状态 `context-stage1-selection-chain-v1-status.json`。

缓存等待连接失败只在600秒后重试同一个docker wait，不能据此重启容器。
launcher自身观察失败会保留已有created/launch文件并停止链，需要核验原handle后接续。
两条新驱动均有独占锁，NPU launcher继续使用统一evaluation/training锁。

全训配置保持DP4、global128、micro8×4、BF16 LoRA和浅层概率头，
ranking/binary NLL各1，lr1e-4，完整clean学习40513条两轮634步，
第64/128/…/576/634步保存。保存完整FP32 master/Adam和联合BF16策略。
全量collector核验所有实际batch顺序/监督、恢复step2状态报告、每rank更新报告、
每个保存点的Adam状态与完整联合策略，统计热身后普通步及首末32步的p50/p95。
它不会重新计算全部processor或反向传播，不把运行自报告当独立重算证据。

## 评测入口

`launch_context_stage1_selection_v1.py --step N`支持初始化0及所有预定保存点。
`prepare_context_stage1_evaluation_v1.py`生成该launcher；本轮step0已prepare-only，
尚未占用NPU。依赖完整训练collector时，从其已核验本地联合策略固定源身份。

同一NPU租约内按顺序执行：

1. 同一源策略256学习音频HF/vLLM推理；不更新参数、不消费标签。
2. 比较全部129个执行张量（128个LoRA/head加冻结marker）、native adapter及
   全256×60概率和输入身份；原策略与重导出策略可有不同序列化ID，张量须完全相同。
3. 完整selection9226音频、完整60候选、直接语音输入，通过既有vLLM入口评分。
4. `collect_context_stage1_selection_v1.py`独立收集并核对源张量、parity、
   全人口请求/输入receipt/逐条预测，重新计算selection指标；不重新生成processor张量。

全量训练前后比较以这个统一context协议为准，不混用旧Final60 CE或Omni不同协议。
仅step634与0自动排队；中间保存点仍可评测，不能称已生成完整RL曲线。

## 本轮验证与尚欠工作

新脚本及嵌入的远程runner编译通过，step0配置prepare-only通过；
既有指标、checkpoint、全人口日程15项测试通过。
这些不等于新自动链端到端NPU验证；实际全训/选择集尚待执行与独立收集。
完整在线RL策略更新、效率profiling和microbatch对照、学术基线及跨数据集迁移仍未完成。
