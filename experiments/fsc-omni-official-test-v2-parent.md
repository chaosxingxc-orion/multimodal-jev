# FSC 冻结 Qwen3-Omni 父提示：完整 TEST 结果

父提示完成官方 TEST 的3793条录音、每条6种相关视图，共22758次端到端音频调用。
严格JSON消费协议下，native正确1752/3793（46.1904%），六视图全对9/3793
（0.2373%）。这是零任务训练的冻结Omni基线，不是任务训练后的学术分类器复现。
完整p1对照尚未完成，不能据此声明提示优化提升。

## 固定协议与证据

- 模型：本地Qwen3-Omni-30B-A3B-Instruct，直接音频与任务schema输入，无转写串联。
- 镜像：`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`。
- 实际容器：`ae16d2e146ecab6207ac5779b6b561336a6d8ef941a83e4e279c9fde4526ebb0`，
  NPU4/5、16 CPU/128 GiB、8 GiB共享内存，已退出0，无OOM。
- 源码：`553c27d823c7e2209fa4041ba7632e5e9d3e3d02`；外部逐文件核验729份源码。
- TEST协议SHA：`55b0e6def7b629dfe5e8912450f00d2218930000291b0866e48b64d5c779a3bf`。
- 生成：BF16、TP2、greedy、最多128输出token、4096上下文，单次replicate。
- 全程40276.47秒（11.19小时），输出217240 token，0次长度截断，0次权重更新。
- [独立核验](../docs/engineering/frozen-qwen3-omni-fsc-parent-test-v2-verified.json)
  验证实际容器、完整覆盖、请求/目标/预测哈希、冻结选择协议及原生标签映射；
  从raw text独立解析JSON、复算逐视图和配对指标，与原score完全一致。

结果根目录：`/mnt/cephfs/xingchao/experiments/multimodal-jev/frozen-qwen3-omni-official-test-v2/parent`。
模型文件前后验证通过属于原运行证据；独立评分没有重新执行模型推理。

## 六种视图

| 视图 | 正确数 / 3793 | 严格正确率 | JSON格式有效率 | schema有效率 |
|---|---:|---:|---:|---:|
| 原生标签选择 native | 1752 | 46.1904% | 50.6723% | 50.6723% |
| 候选重排 permutation | 2037 | 53.7042% | 57.3688% | 57.3688% |
| ID重新绑定 rebind | 1768 | 46.6122% | 52.2278% | 52.2278% |
| 正确候选缺失 missing_candidate | 36 | 0.9491% | 47.5086% | 39.5729% |
| 命题为真 noul.True | 2561 | 67.5191% | 74.4793% | 74.4793% |
| 命题为假 noul.False | 2291 | 60.4007% | 82.1777% | 82.1777% |

严格评分要求原始响应能解析为只有`value`及可选`confidence`的JSON对象；
不剥离Markdown围栏、不从解释中抽取答案，也不把非法格式当作合法null。
目标类型严格区分字符串、布尔值与null，missing_candidate正确结果为JSON null。
该表反映任务判断和消费格式的联合成功率，不能解释为ASR识别率。

## 配对约束

| 同一录音要求 | 同时正确数 / 3793 | 比例 |
|---|---:|---:|
| native与permutation | 1382 | 36.4355% |
| native与rebind | 1410 | 37.1737% |
| native与missing_candidate | 19 | 0.5009% |
| noul.True与noul.False | 1729 | 45.5840% |
| 全部六视图 | 9 | 0.2373% |

六种视图共用同一批3793条录音，不能把22758次调用当成独立样本量。
单独的native与rebind平均正确率相近，不意味着同一录音的决策保持正确；配对
指标是这个任务定义的一部分。missing_candidate的低成功率说明当前严格接口下
缺少正确候选时的未知输出尚不可靠；格式失败与语义失败都计入完整分母。

## 与已有学术基线的关系

SpeechBrain公开任务权重的本地完整TEST复测为3774/3793（99.4991%），
见[学术基线锁定](../docs/research/07-academic-baseline-lock.md)。该结果使用FSC
监督训练的direct模型；本次Omni没有FSC权重训练，并增加严格JSON候选接口。
两者可在横向表中披露，但训练条件与输出协议不同，不能把差距全部归因于骨干
模型，也不能把本次Omni结果写成SpeechBrain训练过程复现。

父提示和p1的比较则使用相同模型、输入、完整TEST、解码及严格scorer。
p1在TEST之前已通过冻结选择协议确定；原队列在父提示成功后启动容器
`6f6fe878a1cbf6906cc679a5f2abcdb6403075cffc23b692f90bb7dbb03d957a`。
该容器已实际inspect为running，尚无完整评分。不得使用本页TEST反馈修改候选、
提示或评分规则；后续提升结论需等待两边完整结果和逐录音配对比较。
