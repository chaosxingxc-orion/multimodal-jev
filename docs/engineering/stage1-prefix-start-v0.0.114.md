# 新前缀布局 Stage1 启动与验收

本轮目标：原始 admitted initializer → 新前缀布局 → 全量 clean TRAIN InfoNCE。旧 Stage1/Stage2 endpoint 不是本轮初始化。Stage2 的 reward/margin 候选不阻塞本阶段。

- 175 物理卡 4–7，DP4，BF16 执行、FP32 optimizer/master/loss。
- Qwen3-ASR-0.6B，q/v LoRA r4 alpha8，共享 Transformer decision head。
- `global_batch = 64 * DP = 256`，microbatch8，40513 条 TRAIN，2 epochs，318 updates，warmup16，peak LR1e-4。
- 纯 candidate InfoNCE，temperature1，probability coefficient0。
- 布局 `jev-stable-prefix-audio-readout-v1`：规则和完整候选定义在音频之前，全部 decision readout 在音频之后。缓存身份和 optimizer checkpoint contract 均绑定布局。
- 正式检查点 `0,32,64,96,128,160,192,224,256,288,318`；独立完整曲线另行评测。

输入发布复用经过认证的旧 acoustic feature，不复用旧 token 布局。`tools/jev/relayout_context_learning_cache.py` 通过同一实际 processor 重建文本模板，逐条比对旧 token/位置、缓存文件 SHA 与张量身份，按 token 长度共享模板编译；32 个由长度排序确定的代表样本（含极值）额外执行真实音频前处理并逐张量比较。发布前要求全40513条完成、上下文无截断。输入缓存只读，热输出在本地持久盘。

`tools/reports/context_stage1_start_pipeline.py` 统一串接现有 prepare/launch/collect/audit：缓存精确容器终态 → 两步 NPU finite → 独立审核 → fresh-process replay → 独立审核 → true-tail → 独立审核 → 正式包与训练。每一步失败即结构化退出，不调用模型，不重启既有实验。部署地址与路径由 `.local/config` 宏文件绑定；共享模板为 `configs/portable/operations/context-stage1-start-pipeline.json`。

固定镜像 CPU 证据：`context-stage1-prefix-cpu-v0.0.112/report.json`，63 passed/0 failed/0 skipped。收集器旧 expected_tests62 与新增测试数不一致，终态代码0与JUnit63已保留；新 pipeline 显式要求63，不把脚本数量错误解释成训练/模型失败。

启动命令（宏文件由本地机器提供）：

```bash
PYTHONPATH=src:.:tools/jev python -B -m tools.reports.context_stage1_start_pipeline \
  --config configs/portable/operations/context-stage1-start-pipeline.json \
  --local .local/config/context-stage1-start-v0.0.117.json --detach
```

历史失败只读保留到当前继任验收完成：v112 CPU缓存缺TRITON_CACHE_DIR；v113缺发布审核文件映射；v114逐样本重复编译完整候选导致CPU瓶颈，受控停止并由v115按音频token长度编译复用替代。对应最小证据在operations下同名目录launch与远端terminal。

当前状态必须现场读 controller / exact Docker inspect；本文不是 live 状态证明。验收要求正式训练进程运行且4rank完成真实更新1、2、3；工程prefix的loss不计入这三组。

追加修复：v115控制器将可更新progress误用不可覆盖commit_json，v116改为atomic_json；v116在/home生成运行宏触发10%剩余容量保护，v117将生成宏集中绑定/srv本地持久盘，未降低存储门槛。缓存v115已发布，全40513条、32条fresh parity、tokens2818–3115，耗时645.1秒。finite与fresh replay均已独立通过，正式状态另以现场证据为准。

## 正式启动验收结果

正式容器 `0a446faed3c8fabc1ed3c508f98e0921babd61f3546f30533fb3f51654ece2fc` 在采集前三步后 inspect 为 running。前三步全局InfoNCE为 4.09342350, 4.09476930, 4.09184140，每步4rank完成，global256，缓存每步命中256/256。统计按各rank8个microbatch贡献之和再跨rank平均，与actor/pg_loss口径一致。前三步不是收敛/收益结论。

证据：`/srv/multimodal-jev-xingchao/operations/context-stage1-start-v0.0.117-observation/summary.json`；正式配置宏位于 `/srv/multimodal-jev-xingchao/operations/context-stage1-start-v0.0.117/context-infonce-prefix-formal-v0.0.rc6.json`。控制器继续运行完整318步并独立collect，正式全程审核与评测另行执行。
