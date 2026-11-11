# 真实语音四卡第二步恢复准备

当前模型仍为 Qwen3-ASR-0.6B。原四卡真实语音试验只完成一次更新，
`probe_span_fsdp_npu.py` 明确拒绝 optimizer continuation；合成模型的恢复
测试不能替代真实语音恢复。本次新增独立入口，未修改正在运行的三方法实验
及自动评估队列所固定的任何文件。

`probe_span_fsdp_resume_npu.py` 只接续已独立验收的
`span-fsdp-audio-npu4-v2` 和 `span-fsdp-audio-suite-v1`，不从单卡 Adam 迁移：

- 先重新核验原四 rank 更新、原始源码/文件身份、导出 adapter、48 条 TP1 gate。
- 加载同一 BF16 adapter 后分片，核对每个 FP32 master 降为 BF16 与该本地分片一致。
- 校验全部参数顺序、shape、rank/world size、Adam 超参、完整有限 FP32 moments
  和 step=1，全部校验通过后才写入；保留低于 BF16 分辨率的 master 信息。
- 保存恢复后的真实 master/Adam tensors，供独立核验逐元素对照原断点。
- 全局批次前进四个互补样本对：四个 rank 分别从 pair 2/3/4/5 前进到 6/7/8/9，
  每卡两条、每项分类一正一负。不能用单卡的 `next_batch_index=previous+1`。
- 完成下一次真实 verl 更新，保存原生 Adam step=2 和未合并 BF16 adapter。
  下一版本仍需独立 HF 精确重载、48 条完整候选及联合动作 TP1 gate。

本地 17 项测试通过，包括恢复后下一步 Adam 与未中断 CPU 对照精确相同，
以及拓扑/参数/状态/数据游标损坏的拒绝测试。另使用四卡真实试验保存的
448 份 master/Adam 分片做 CPU 恢复，全部通过。

固定镜像 `sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`
中的独立 CPU 容器 `4d9bb77ae78249809f5036542869ab2200f8b02d5945ca4396d4db3335f14001`
退出 0；未映射 NPU、无网络。该复核使用镜像原生 optimizer metadata，
没有宿主兼容性改写，448 份实际保存的分片/Adam 恢复全部精确相同。
证据见 `span-fsdp-resume-prepared-v1.json`。

已准备独立源码快照、config 和四卡启动 argv：
`/mnt/cephfs/xingchao/experiments/multimodal-jev/span-fsdp-audio-resume-prepared-v1`。
**尚未派发 NPU 恢复**：4–6 卡上的既有训练/评估及其自动评估队列需先正常完成，
启动前核对 4–7 卡均可用，再记录真实 launch/container identity；不能把 prepared
状态当作已启动。不要覆盖原训练根目录或修改固定快照。

CPU 检查使用重建的本地张量布局，未验证实际 FSDP materialization、HCCL、
真实语音第二步更新和跨后端概率。即使后续第二步全部通过，也仅证明固定四卡
拓扑恢复，不证明未中断/重启的完整 NPU 轨迹等价、任意拓扑恢复、TP4 推理、
Ray trainer 或学术收益。原生 SLURP 学术复现、方法收益和跨任务泛化仍为独立目标。
