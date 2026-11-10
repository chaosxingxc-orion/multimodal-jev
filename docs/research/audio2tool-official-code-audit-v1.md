# Audio2Tool 官方代码与论文评分协议核查

核查日期：2026-11-10。官方项目页面现链接到
[audio2tool/Audio2Tool](https://github.com/audio2tool/Audio2Tool/tree/b9d5383ca30945fbafb965a04bea40117e1e8a82)，
固定 commit `b9d5383ca30945fbafb965a04bea40117e1e8a82`。
代码 LICENSE 为 MIT，数据集仍按其独立 CC-BY-NC 4.0 许可处理。
这解决了“官方代码未定位”的问题，但尚未证明当前代码就是论文 v1 的历史实现。

## 评分差异

| 项目 | 当前固定官方代码 | 对横向比较的影响 |
| --- | --- | --- |
| 多工具调用 | `metrics.py::compute_multi_tool_metrics` 将工具列表转为集合；首工具另行比较 | `all_tools_match` 忽略顺序和重复次数，不等于论文锁定的 ordered sequence accuracy |
| 参数类型 | normalization 将值转换为小写字符串 | 字符串 `"1"` 与数字 `1`、`"true"` 与布尔 `true` 可相等；不能当成类型严格的 Jev 正确率 |
| 参数 F1 | `evaluator.py::_compute_metrics` 对逐样本 F1 做 `np.mean` | 不等于汇总 slot 计数后的 micro-F1 |
| 总体 EM | 主工具匹配且主参数匹配 | 不能自动解释为完整多调用及全部参数的有序 EM |
| 错误分母 | 聚合前排除 `error is not None` 的结果 | 失败样本可能不进入分母；须额外报告完整覆盖与全分母结果 |

已实际导入下载的官方 metrics 模块运行人工反例：颠倒两个工具仍 `all_tools_match=True`；
漏掉一个重复工具仍为 True；上述类型转换判等。一个样本正确 1 个槽、另一样本漏掉
全部 3 个槽，逐样本平均 F1 为 0.5，汇总 micro-F1 为 0.4。
反例记录见 [metric fixtures](audio2tool-official-metric-fixtures-v1.json)。
聚合 evaluator 仅静态审阅，未运行完整数据评测。

## 输入、提示与数据接口

Evaluator 将 `gt_query=sample.query_text` 传给模型。已检查的
`qwen3_omni_vllm.py` 两种 HTTP payload 构造路径均只取音频、工具定义、system
和固定请求文字，不把 `gt_query` 放进 payload；不能仅因 evaluator 传参就断言此客户端泄漏。
这不证明其他客户端安全，正式音频适配应直接阻断参考转写进入 actor。

该客户端默认提示要求始终调用工具、缺参用默认值、不得澄清，与 Jev 的缺失/未知状态
不一致；不可直接当作 Jev 提示。默认请求可重试且工具 API 失败可回退，不能原样用于
本项目要求保存失败、禁止自动重放的正式推理账本。

`release_dataset.py` 期待 `metadata.jsonl` 加音频目录；已固定 HF release
`f1388da9a3189541ab82adac88824a0661670c43` 使用 `public/tierN_*_data/*.json`
及音频目录。其他 tier loader 存在，但本次未逐一验证，不能声明当前默认 loader 直接兼容。
客户端还有音频重采样和 PCM16 编码步骤，正式接入须固定并审计音频变换。

## 比较协议

保留论文 v1 指标定义，另登记当前官方代码指标；两者不得混用名称或静默替换。
正式比较前需完成固定 release 的 loader/候选供给审计、历史 prompt/checkpoint 对齐，
以及完整样本覆盖、错误分母和多调用参数绑定控制。当前没有新增 Audio2Tool 模型成绩。
Audio2Tool 全部仍作为外部 TEST，不用于教师反馈、训练或提示选择。

[来源请求与 SHA](audio2tool-official-source-audit-v1.json) 记录官方页面、commit、树和源码。
本次只读元数据/代码并运行人工指标反例，未加载模型或读取数据集答案。
