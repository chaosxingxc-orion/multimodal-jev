# SLURP HuBERT官方权重复测：离线资产缺口

本轮实际读取共享模型目录元数据，共1,333个唯一文件，无读取错误；其他指定的本地
HF缓存、omni/models、nga/models目录不存在。按精确大小筛选，不依赖文件名，因此
可识别改名或HF内容寻址文件；没有相同大小的候选需要进一步hash。未遍历symlink目录，
未搜索175/181本地盘或其他未指定目录，因此不宣称全基础设施都不存在。

来源为已固定的[官方接口审计](slurp-hubert-interface-audit-v1.json)。可执行清单包含
仓库revision、精确文件大小、内容校验值及本次路径覆盖，见
[资产报告](slurp-hubert-offline-asset-inventory-v1.json)。

| 依赖 | 字节 | 本次扫描 |
| --- | ---: | --- |
| 任务hubert.ckpt | 377575130 | 未找到 |
| 任务model.ckpt | 32722503 | 未找到 |
| tokenizer_58_unigram.model | 238221 | 未找到 |
| 默认HuBERT初始化pytorch_model.bin | 377569754 | 未找到 |
| 合计 | 788105608 | 约751.6 MiB |

任务revision为`6644fb49e209d2c4c3ea6bfd8accbe06b2df28c4`，初始化模型revision为
`dba3bb02fda4248b6e082697eee756de8fe8aa8a`。大权重按LFS实际内容SHA256验收，
非LFS tokenizer按Git blob SHA1（含blob长度前缀）验收，不混用blob ID与普通SHA1。
LFS pointer文件不能代替实际权重。当前只有配置、接口及来源元数据，不能加载模型复测。

工具 `tools/reports/audit_slurp_hubert_local_assets.py` 不调用网络或模型loader，
3项检查验证改名文件、同大小错误内容、Git blob计算、LFS pointer拒绝和未遍历路径边界。
本轮未更改共享文件、下载权重或安装依赖。遵循当前不新增模型下载的执行约束。

资产齐备后的既定路径是隔离环境中调用固定官方接口，先单条TRAIN/batch1检查，
再按冻结原生协议复测；不在RL镜像内覆盖SpeechBrain/Transformers版本。当前默认配置
会先加载基础HuBERT再恢复任务权重，不擅自省略前者而称完整默认配方已复现。
历史训练commit与发布配方不一致的问题仍保留；官方权重复测不能替代历史训练重现。

已完成FSC官方权重全3793条复测（3774正确）与已有Omni clean原生结果继续保留。
SLURP HuBERT作者报告87.54% intent、75.10 SLU-F1仍为文献值，不能填入本地结果列。
本次清单是复测准备，不是模型结果，也不阻断当前Qwen-ASR全量GRPO/RLOO实验。


## 175/181可访问本地盘补充扫描

本次远程只读检查`/data/xingchao`、`/data/models`以及指定HF缓存目录，复用原工具
按精确大小筛选、匹配后内容hash。175检查342071唯一文件，181检查11875唯一文件，
两机均无同大小候选，四项固定资产仍未找到，没有下载、加载模型或修改远程文件。

覆盖不完整：两机`/root/.cache/huggingface`不可访问；175另有873项目录访问错误，
181有30项，主要为容器所有的运行缓存。未遍历symlink目录，未扫描Docker overlay，
不能声称全机器不存在。第一次扫描在不可访问root处退出，没有产出成功收据；
随后显式分离不可访问root后完成可访问范围，完整错误清单保留于结果。

证据：[175扫描](slurp-hubert-offline-assets-npu-v1.json)、
[181扫描](slurp-hubert-offline-assets-npu2-v1.json)。
当前官方SLURP HuBERT复测仍缺788105608字节已固定资产；文献值保持文献身份。
这不阻断当前GRPO/RLOO及已有FSC官方权重参考，后者本次归档、模型及3793音频复验通过。
