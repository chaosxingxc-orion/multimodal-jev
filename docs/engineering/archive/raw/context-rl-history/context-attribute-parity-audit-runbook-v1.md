# 原生属性六请求NPU诊断与独立验收

当前候选为 `.local/decision-objective-research/context-attribute-parity-npu-v2-prepared`。
原v1存在请求哈希序列化不一致，保留取证，不派发。v2固定镜像CPU的14项检查及六请求
processor已通过；仍没有真实紧凑请求的Qwen/NPU概率结果。

## 执行范围

三个已固定TRAIN音频分别以Direct60和Q124输入，共六请求；固定Stage1终点、BF16、
同一共享浅Transformer head。HF与vLLM分进程执行，概率须逐候选完全一致且不饱和为0/1。
只验证这六个输入的推理，不包含DP更新、新进程optimizer恢复或完整人口/效果准入。

已有GRPO/RLOO及迁移评测队列优先。当前启动器要求MINDS队列已完成，再获取既定
设备租约并验证卡7实际空闲。普通观察至少600秒，准确docker wait不重建容器。
退出码非零先收集失败现场；观察失败不代表终态，不重复派发。

本地只读核验命令：

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:tests:. python -m tools.reports.launch_context_attribute_parity_npu_v2 \
  --prepared .local/decision-objective-research/context-attribute-parity-npu-v2-prepared
```

设备及队列条件满足后，同一命令加`--launch`。启动入口保存attempt和容器ID后才start，
持租约等待原容器终态并收集。当前没有自动新增此队列，也没有实际NPU启动记录。

## 独立验收

实际收集完成后使用下面的离线审核命令；不重新推理、不读取TEST：

```bash
PYTHONPATH=src:.local/verification-python:tools/jev:tests:. python -m tools.reports.audit_context_attribute_parity_npu \
  --prepared .local/decision-objective-research/context-attribute-parity-npu-v2-prepared \
  --launch .local/decision-objective-research/context-attribute-parity-npu-v2-launch.json \
  --inspect .local/decision-objective-research/context-attribute-parity-npu-v2-inspect.json \
  --collection .local/decision-objective-research/context-attribute-parity-npu-v2-collected \
  --transport .local/decision-objective-research/context-attribute-parity-npu-v2-collected.transport.json \
  --cpu-evidence docs/research/context-attribute-parity-prepare-fixed-image-cpu-v4.json \
  --cpu-collection .local/decision-objective-research/context-attribute-parity-prepare-cpu-v4-collected \
  --output docs/research/context-attribute-parity-independent-npu-v2.json
```

审核绑定实际镜像、命令、环境、设备、只读挂载、终态及完整传输hash；重建六请求的
audio/catalog/候选顺序，再将两端token、mask、BF16特征和读出位置绑定到此前实际CPU
processor结果。两个后端即使一致，也不能消费与锚点不符的输入。概率重新比较，不只信
runner写出的parity摘要。收集包中的终态还必须指向同一执行文件清单。

## 当前验证证据

12项本地检查通过：真实封存包和CPU输入，模拟模型概率、容器及传输记录。
覆盖完整审核和双方相同错误输入、读出位置、面板、设备、挂载、环境、镜像、非终态、
传输缺失、错误终态与profile。首次两例失败是测试夹具共享参考对象；独立复制后通过，
没有降低审核标准。证据`docs/research/context-attribute-parity-independent-auditor-readiness-v1.json`。

这完成了有限诊断的离线审核入口，不代表实际NPU推理成功，也不成为当前简单任务的新门槛。
