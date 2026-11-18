# MINDS真实模型声学逐层诊断

2026-11-17 UTC。继卷积局部诊断后，直接捕获实际HF与vLLM声学路径。当前结果不是正式训练准入。

## 冻结输入与路径

使用原声学KV16 MINDS parity包作为父包，manifest为`b0a8b36e9833c90a7e4f7a0eb47f9aa2f0f0554a828d2ebc5587f86e25d9b858`。沿用同一Stage1策略、模型、候选定义、原始缓存和镜像。诊断隔离在175物理7，热状态仅本地盘，缓存和模型只读。

按padded mel帧数1100、2600，各取冻结回执中字典序第一ID，不用gold和预测选择：

- `minds14:005f56c5e87a7f85a72fd7e33beedfb966742d4657aeb2b0d32b3e7593f5d720`
- `minds14:453e3eb7c3f2a3cde478db459202d62d5bf3ea26e8d32fc0d1300d685136f2ca`

顺序执行三个独立进程：HF actor profile、原单录音声学worker、声学批处理worker。推理路径采用service batch32/KV16配置；每条先单独请求，再一起提交两条。新音频缓存UUID确保批次比较会实际执行声学编码。

捕获内容：位置缓存、conv2d1/2/3、conv_out、encoder输入、18层encoder输出、ln_post、最终声学投影。保存独立CPU tensor快照，钩子不替换输出。记录最终logit/prob，但本实验不作目标评分。

## 已有证据

v1：HF完成；baseline单条捕获完成，在批次前因遗漏`measure_context_acoustic_candidate.py`导入依赖而失败。原容器exit1，完整失败日志和传输材料已保留，传输hash重验通过。不能将v1称为三路径成功。

v1可用的两个单条捕获中，HF与baseline从conv2d1到最终声学投影全部exact，包括18层encoder。HF位置buffer尺寸13×896，vLLM预留1500×896，完整buffer尺寸不同不能直接判为数值差异；实际encoder输入相同。

选择完成后与旧失败记录交叉核对：1100帧样本原service32结果exact、声学KV16结果logit误差0.15625；2600帧样本两个旧运行均为0.125。这说明有限选样覆盖了真实失败现象，但不能用该核对反向调节样本选择规则。

v2补齐辅助模块，以新包/新容器执行，原v1不修改。执行终态须以当前句柄和收集报告核实，不据本文件判断进程存活。

## 入口与证据目录

- 共享配置：`configs/portable/operations/context-audio-stage-probe.json`，宏`AUDIO_STAGE_PROBE`。
- 生命周期：`tools/reports/context_audio_stage_operations.py`，prepare/validate/launch复用既有排他租约、设备空闲、挂载、固定镜像、精确容器句柄及收集检查。
- 执行与独立比较：`tools/reports/context_audio_stage_probe.py`。
- 只读钩子：`tools/reports/context_audio_stage_capture.py`；worker扩展：`context_audio_stage_worker.py`。
- 本机绑定：`.local/config/minds-audio-stage-v2.json`。
- operations根：`/srv/multimodal-jev-xingchao/operations`，运行子目录`minds-audio-stage-v1`、`minds-audio-stage-v2`。
- 测试：`test_context_audio_stage_capture.py`检查快照不受原位修改污染、移除钩子有效；`test_context_audio_stage_comparison.py`检查独立重算能检出单层单元素BF16扰动。两项通过，不代替NPU结果。

## 解释边界

先比较同一输入的真实中间tensor，再判断声学实现、语言调度或head的责任。单条exact不能推断多条exact；两条批次exact也不能推断259条面板通过。不能改容差、删失败样本、用top1未变绕过概率一致性，也不能据局部结果提前开始正式训练。

## v2完成与结论

v2已exit0，完整收集后在本地重新加载tensor，独立重算报告一致。manifest `22ef050e0e86780f9e548628148ed7db3c524638c7f2e1d33f8f753024c50143`；report SHA256 `e3e3519c117d212e690f578907f1179ef108139bf052ade63c2aa8bdd4a22946`。审核状态`stage_tensors_independently_compared_not_admission`。

两个样本在baseline和acoustic的单条路径中，所有声学计算输出与HF exact；两条组成批次时，从encoder输入、18层输出到最终声学投影也全部exact。四组最终logit比较（baseline单条、baseline两条、acoustic单条、acoustic两条）最大误差全部为0。位置缓存完整尺寸不同仍单独记录，没有将其截断后伪称完整buffer相同。

这不支持持续性LoRA加载错误，但不能排除更大批次的LoRA执行数值或其他语言层问题。vLLM日志中processor目录的`Unrecognized model`出现在初始化阶段，随后模型加载与推理成功；不能单凭该日志认定adapter加载失败。启动校验adapter文件身份与快照，策略切换另核验实际q/v张量、缩放和槽位；文件身份正确不证明任意batch执行exact。

下一步：重建原259条面板按macro128、rollout32及service排序得到的真实失败批次，保留原邻居和顺序，捕获首个差异层。不能根据两条批次结果放行全部面板。
