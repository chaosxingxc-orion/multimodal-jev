# 固定脚本数据采样

入口为tools.sample_dataset，编排kind为dataset_sample，复用tools.run_operations及
现有宏解析、存储检查、发布清单和文件锁。不调用模型、不创建定时器或训练任务。

从configs/dataset-sampling-macros.example.json取所需宏加入自己的本地绑定文件；
共享模板为configs/portable/operations/dataset-sample.json与dataset-sample-pipeline.json。
在仓库根目录运行：

```bash
PYTHONPATH=src:. python -m tools.run_operations \
  --config configs/portable/operations/dataset-sample-pipeline.json \
  --local .local/config/macros.json
```

输入必须是绑定SHA256的JSONL对象记录；id、split、label默认均为非空字符串，
字段名可在新版本模板配置。所有行必须属于声明的train或learning，重复ID、缺标签、
混入其他split、请求数量超过人口都会拒绝。此检查约束记录声明，不能识别被错误
改标为train的测试数据；上游数据发布仍负责真实划分和来源审核。

uniform按seed与ID的SHA256排名选择；balanced对固定顺序的类别轮流取样，少数类
耗尽后从剩余类别补足，不放回、不伪造样本。相同seed、人口及ID不受输入行重排影响。
balanced改变子集类别分布，适合明确登记的调试或采样任务，不替换现有全量训练日程，
也不能用于构造有偏的正式评测集。

输出sample.jsonl保留选中行的原始内容（末行缺换行时补换行），sample-manifest.json
记录ID、seed、总体/子集类别数量，COMPLETE.json绑定输入、配置、源码及输出hash。
成功重入验证身份与产物后返回；失败保留现场，不自动覆盖或重试。当前实现会将人口
载入内存，适合现有数万级标注记录；不宣称流式处理海量数据。

采样和编排联合25项测试通过。旧local-operations-v1锁保持历史身份；修改前的
run_operations.py已按旧hash保存至configs/releases/local-operations-v1/source/tools/。
当前采样增量由configs/releases/local-operations-v2锁定，不把旧锁在新源码上视为通过。
研究产物保留清理、远程收集的配置归并仍待完成，此交付不代表全部日常任务已闭合。
