# 双信号训练效率的匹配比较

目的：用实际完整训练步骤判断 batch4 的收益，支持用户授权的旧训练替换。
本次只准备对照包和统计脚本，未执行 NPU，也没有加速结果。

两个包从已通过固定镜像依赖验收的整合 v3 生成。固定相同镜像、源码、Stage1
起点、SLURP 日程前两个 global128 batch、micro8、DP4、joint RLOO 目标和每步
LoRA/head checkpoint。service token 预算均为 16384；唯一计算配置差别是
max_batch_queries=1 或 4，两个实例各有独立输出和初始为空的 reference cache。
不能让后一组消费前一组缓存后把收益全部归因于合批。

两个包各 398 文件，已检查全部实际 hash 和指向包内的 pins。序列化包身份：

- serial：`d820c90987f590aea3ebdfcbca0458e18a0878d1be5db946111abc4443b41d90`
- batch4：`a9de2f0e97931ae70b5ebc7dc88f013f23dd2c4bdce62c375c02af303e75a3ba`

准备命令（本地文件绑定机器路径，模板不绑定本机目录）：

```bash
PYTHONPATH=src:. python -m tools.reports.prepare_context_dual_throughput_pair \
  --config configs/portable/operations/context-dual-throughput-pair.json \
  --local .local/config/dual-throughput-pair-v1.json
```

已存在输出不会被覆盖。`PAIR.json` 记录两臂身份，所有 dispatch/switch 标志均为 false。

结果读取入口是 `tools.reports.compare_context_dual_throughput_pair`，共享模板为
`configs/portable/operations/context-dual-throughput-compare.json`。只消费独立 collector
生成的终态 transport 与实际归档，校验被读取证据的 hash、执行包身份和正常退出。
每一步取四 rank 的最大 total_seconds；包含输入、reference、behavior、actor、
checkpoint 和激活。它核对参考缓存 miss、完整 collective parity 记录、激活 policy ID，
并逐 rank 读取两步 checkpoint 比较参数、Adam 和 contract 的精确一致性。
初始加载、模型构造和数据集初始化不在每步 timer 中，需另外报告作业 wall time。

11 项本地测试通过，包含完整封包差异限定、错误 hash/路径拒绝、最慢 rank 计时，
以及不完整 rank、非法 timer、不同缓存条件、parity 失败和 Adam 变化的拒绝。
测试使用合成计时，不是实测速率；没有生成伪造的实际比较报告。

这两个前缀用于筛查，不足以单独证明持续训练吞吐。batch1 也使用新版 segmented
runtime；它和旧运行的数值一致性仍由 serial/batch 音频 probe 检查，不能混为同一实验。
正式替换还需要独立完整更新审计、新进程恢复，以及覆盖代表性长度/缓存条件的计时。
更换目标函数时应作为新实验从固定 Stage1 起点开始，不能把旧 joint-action RLOO 的
Adam 状态直接续到双信用目标并拼接成同一学习曲线。

本轮资源检查：181 的 0–7 均由已有 TP4 服务占用；175 的 0–3 为已有 TP4 服务，
4–7 为当前 RLOO。没有停止这些服务，也未假定瞬时计算利用率为零代表空闲。
有限 NPU launch/收集、训练 audit 与恢复封包入口现已接好，probe 已增加显式
idle_resource_window 模式，仍需排他锁及设备确实空闲。完整审计路径仅通过 CPU
合成归档检查；实际 NPU 测量和受控交接仍待，不可绕过锁而抢占设备。
