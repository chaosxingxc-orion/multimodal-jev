# 五个核心数据集的音频文件哈希交叉核查

在既有FSC/SLURP/SFC完整native release核查基础上，新增Audio2Tool全部
36,419个音频资产、16,843个tier/query组合，以及MMAR完整1,000条输入。
共18个划分/层级，153对比较，全部共享文件哈希数为0。其中跨数据集116对，
同数据集不同split/tier为37对。结果见
[核查记录](../../../external-audio-overlap-v1-result.json)。

输入清单SHA、完整条数、唯一ID/路径、音频哈希格式均核对。MMAR每条输入
的音频元数据须与完整inventory一致；Audio2Tool逐tier数量须与已固定的
inventory结果一致；三个native release重新执行已有完整性检查。
随后用另一套直接构造集合的代码独立重算全部153对交集，与报告一致。

这次使用既有音频文件SHA，没有重新解码或重哈希波形，没有读取独立targets、
MMAR references或Audio2Tool query标注，也没有调用模型。manifest中既有
标签相关说明仍可能被解析，所以不声称完全没有接触任何标签元数据。

统计单位必须分清：FSC/SLURP/SFC为native请求，MMAR为问题；Audio2Tool
为音频资产路径，包含tier7的agent回合，既不是native推理请求数，也不是
source query数。报告保留两侧单位，不把它们相加作为评测样本分母。

零交集只排除当前固定清单之间完全相同的文件字节，不排除转码、重采样、
裁剪、相同来源视频、相同语义/文本或预训练暴露。它不是“无数据污染”或
“未见数据集泛化”的证明。该核查不改变已冻结的划分、提示或测试评分；
后续跨任务结果仍需同初始化/预算对照及明确的开发暴露记录。

完整报告和本次执行脚本保留在
`/mnt/cephfs/xingchao/experiments/multimodal-jev/external-audio-overlap-v1/`。
本核查没有重跑已存在的推理实验。
