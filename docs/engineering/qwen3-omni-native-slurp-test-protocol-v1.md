# 冻结 Qwen3-Omni 原生 SLURP 完整 TEST

固定官方 real TEST 的13078条录音、2974个语义组，不抽样、不读目标调提示。
使用已完成 TRAIN16 探针的原生复合输出指令、taxonomy、BF16、seed17、
greedy、4096上下文、256输出token和同一官方Qwen3-Omni权重。
输入为音频和完整taxonomy，输出scenario/action/entities。无ASR串联。

原生复合任务与FSC-Jev教师提示优化独立；本次不替换为单value指令，
不把本次TEST反馈传入教师。官方评分以录音为单位，报告joint intent、
entity exact F1、word/char F1和SLU-F1；保留格式失败、重复实体和输出截断。
严格解析，不修复Markdown围栏。全部输出与容器、应用终态成功后才评分；
不完整结果不报告完整TEST成绩。权重前后全文件校验，禁止静默截短音频。

服务器175使用授权的NPU6/7，TP2，沿用固定镜像
`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
FSC选择队列使用4/5，通信端口分开。单次容器最长86400秒，观察86700秒，
由脱离会话的监督进程管理，不自动重试。模型不下载、不训练。

该实验是冻结通用基座的完整原生任务基线，不能替代SpeechBrain任务训练
权重的官方复测，也不能据其成绩宣称Jev跨任务提升。

## 完整运行后的独立验收

`tools/jev/verify_omni_slurp_test.py`在监督进程及应用成功终止后才读取预测。
它核对登记时的精确容器、镜像、配置和Git推理快照，验证与TRAIN16探针的
权重清单、运行版本、提示和解码参数一致；要求全部13078条录音/2974组，
无重复、无遗漏、请求绑定正确，并核对上下文预算和截断计数。
实际Docker配置还必须符合NPU6/7及三个管理设备的精确映射、16CPU、128GiB
内存、8GiB共享内存和host网络；禁止特权模式、额外设备请求、重启或OOM。
容器命令、全部bind挂载和显式环境变量须与保留的launch一致。
语义或格式失败不阻止工程验收，也不被修复。

验收在已有官方评分环境中重新评分，归一化预测必须逐字节相同，整数计数
必须精确一致。官方集合迭代顺序导致的浮点差异使用相对/绝对容差1e-12，
每处非零差异均披露。该容差只用于独立复算核验，不用于判定模型提升。

完整运行结束后，在工程根目录执行：

```bash
PYTHONPATH=src /mnt/cephfs/xingchao/experiments/multimodal-jev/environments/slurp-official-score-v1/bin/python \
  tools/jev/verify_omni_slurp_test.py \
  --root /mnt/cephfs/xingchao/experiments/multimodal-jev/frozen-qwen3-omni-slurp-native-test-v1 \
  --launch-record docs/engineering/frozen-qwen3-omni-selection-and-slurp-test-launch-v1.json \
  --native-probe /mnt/cephfs/xingchao/experiments/multimodal-jev/frozen-qwen3-omni-slurp-native-train16-v1 \
  --output /mnt/cephfs/xingchao/experiments/multimodal-jev/frozen-qwen3-omni-slurp-native-test-v1-independent-acceptance
```

当前尚未对全量TEST执行验收。44项相关测试覆盖完整
数量的合成记录、遗漏/重复/错绑/预算异常、浮点复算与运行前拒绝读取。
其中20项运行配置反例覆盖OOM、错误设备/映射/权限范围、资源漂移、
重启、挂载、命令和环境变量变化。原始运行容器已通过这些配置的只读预检，
证据见[运行预检](slurp-runtime-preacceptance-v1.json)；运行中通过配置核对
不代表推理完成或全量成绩验收通过。
编排测试显式模拟Docker和官方评分器，不代表真实模型或完整测试集成绩。
