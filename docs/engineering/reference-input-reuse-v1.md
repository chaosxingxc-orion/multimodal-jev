# Reference 独立输入校验复用与分项计时

## 范围

开发候选 v0.0.51 延续 v0.0.50 的 journal 内容绑定复用，增加独立输入重建复用。
本轮目标是定位并减少完整 reference 输出审核剩余的 CPU 开销；不改变数据划分、
训练/评测指标、NPU 源码、运行队列或准入条件。

实现：`tools/reports/context_reference_input_reuse.py`，`InputReuseEngine`。
共享配置：`configs/portable/operations/context-reference-input-reuse.json`。
本地宏：`REFERENCE_INPUT_REUSE_CONFIG`。复用已有宏加载器和审核基座。

## 可信边界

复用对象是原 `context_reference_input_evidence.bind` 的纯计算结果，完整输入为：
proof、原始 receipts 字符串、全量有序 rows、完整 task、release hash、processor hash。
所有这些输入和审核代码内容参与键；不按 sample ID、mtime、size 或旧 PASS 命中。
原 `load` 每次重新校验 proof/receipts 的真实字节 SHA256 及二者绑定关系。
数据来源 load、endpoint/checkpoint、processor、reference record 绑定仍每次执行。

一次 `InputReuseEngine` 显式共享同键 Future。原函数执行一次，并发相同请求等待同一结果；
最多一个输入重建在计算，默认保留两个结果。返回深复制，失败不自动重试。
计算后和消费前重新验证内容键；要求调用输入在审核期间保持冻结，不声明原子快照语义。
原源码不 monkey patch；适配器按明确源码锚点替换局部导入，锚点变化立即拒绝。

原日志审核及 v0.0.50 的文件内容验证继续执行。容器/transport/parity 准入在外层，
本轮输出证明差分不能代替整条 Stage2 生命周期审核。

## 测量

每次重放记录完整墙钟耗时及以下独立分项：

| 分项 | 内容 |
| --- | --- |
| endpoint_bundle | 模型 endpoint 执行张量、合同及文件验证 |
| release_population | clean release 来源重建和 population 准入 |
| processor | processor 资产哈希验证 |
| independent_inputs | 原始 receipts 验证、输入重建或内容绑定复用 |
| record_bindings | 独立 tensor 哈希与缓存记录的绑定 |
| journal | 全量 reference 输出日志审核或内容绑定复用 |

分项之外仍有 JSON 读取、hash/population 构造等开销，因此分项之和不等于总时间。
首次和第二次完整结果都逐字段比较原始独立审核证明；保留历史证明 provenance 以便差分，
本次执行 provenance 单独落盘。状态、配置、计时、两份证明、报告和终态均写本地持久盘。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=.local/verification-python:src:tools/jev \
/usr/bin/python -B -m tools.reports.context_reference_input_reuse \
  --config configs/portable/operations/context-reference-input-reuse.json \
  --local .local/config/context-reference-input-reuse-v0.0.51.json
```

证据目录：`/srv/multimodal-jev-xingchao/operations/context-reference-input-reuse-v0.0.51/`。
该入口只执行 CPU 验证，不派发训练。整条生命周期接入必须另行差分验收，不能热替换
冻结 v0.0.49。新候选未取代原候选之前保留原依赖与证据，不擅自删除在用源码。

## 全量验证结果（2026-11-18 UTC）

40513条样本、317个macro，两次完整输出证明均与原独立审核证明逐字段相等；终态exit0。
输入重建与journal各计算一次、复用一次，模型调用0，未执行NPU或完整runtime/transport重审。

| 阶段 | 首次（秒） | 重复调用（秒） |
| --- | ---: | ---: |
| 完整输出审核 | 138.588 | 23.700 |
| 独立输入 | 78.769 | 5.042 |
| journal | 44.909 | 5.059 |
| 数据来源 | 5.164 | 4.995 |
| record绑定 | 3.403 | 3.434 |
| endpoint | 0.936 | 0.397 |

同一候选重复调用比首次减少82.90%；对比上一候选v0.0.50记录的重复调用88.903秒，
本次23.700秒减少73.34%。后者是跨运行单次测量，不是配对多次统计；首次138.588秒
高于上一候选129.125秒，不能宣称冷启动加速。收益针对同一审核进程内共享引擎的重复依赖。
进程重启后缓存丢失并重新计算，不跨进程读取旧PASS。

剩余开销分散在完整文件哈希、数据来源、record绑定和JSON/population处理，已没有之前
独立输入重建的单个约79秒大块。优先开展生命周期接入和完整差分验收，不继续盲目增加线程。
