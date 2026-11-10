# 预训练 Omni 全六条逐层诊断入口

`diagnose_omni_pretrained_layers.py` 将已验证的只读 `PaddingTrace` 接到真实
Thinker text model，保留原 TRAIN6 全部六条首决策及固定 false+EOS。
每条只比较原五条件中的 audio_padding 与 text_padding，保持音频特征公共宽度、
FP32输入特征、CPU BF16 autocast、原始波形边界及模型参数不变。
有效token与位置逐位一致，只有mask排除的左padding不同，不读目标值。

## 运行约束

完整执行须绑定原精确诊断容器
`0de5e886048040b9fa5489586f4a7fcddea88085701bd1818a8a6a94946e36db` 的成功Docker终态，
并核验应用终态、intent/source快照、identity哈希、全部30条记录与已保存logits。
inspection必须由外部协调者实际inspect捕获并固定哈希；文件自身不是独立信任根。
缺少完整依赖时拒绝加载权重。入口不自动调度、不重试，也不修改现有运行。

固定镜像、模型资产与实际HF实现哈希必须与原诊断一致。12次冻结前向保存逐层
有效token张量与比较结果，并核对响应logits是否精确复现旧条件；若不复现，保留
差异，不将其自动归因于padding。每行参考tensor payload上限8 GiB，完整6行上限
48 GiB，另有序列化开销；没有优化器、梯度、生成或任务收益声明。
正式派发前还需登记CPU/内存/时间预算及精确容器，不因工具存在就自动执行。

## 预检查

`--preflight-only` 可在原诊断运行期间执行，仅验证真实processor、完整输入及两条件
的对齐，核验模型文件但不加载权重。它不要求原诊断已经终止，不代表完整执行可派发。
输入检查测试和既有trace异常清理测试共7项通过；固定镜像实际preflight另存结果。

实现源码 `8e7eaca` 的首次预检查容器
`08e3b135494578d6a8f90d6b583aa9e7b8e913fdbf409de4c375aeb7b3bf1593`，
2 CPU/16 GiB/1800秒上限，无网络/设备，模型与数据只读。
源树为 `omni-layers-preflight-v1/source`，完整模型诊断尚未派发。

## 预检查验收与依赖分支测试

上述精确容器现已退出0；独立核对完整源码、模型清单、原六条身份、response、
固定容器资源及两条件结果。六条左padding计数为 `[154,252,0,251,519,402]`，
与已验收原始前向的prompt长度逐条匹配。第三条是无需新增左padding的天然对照。
见[预检查验收](omni-layers-preflight-v1-verified.json)。模型权重未加载。

随后仅增强完整执行的 `complete_dependency` 分支：固定原 intent SHA，
核对实际容器命令、CPU/内存/无网络无设备、身份和source commit，避免重新哈希
改动的intent即可绕过原登记。预检查未经过该完整执行分支；新的分支不冒充已执行
预训练诊断。输入准备和前向代码未改变，既有快照保持原样。

包含成功完整信封、运行中/失败/OOM、命令/资源/identity/source篡改、结果缺行、
权重更新和重新哈希intent的反例；这些依赖测试明确mock logits审计，实际logits
审计另有独立测试。与trace清理、logits审计一共24项测试通过。当前原30条件诊断
尚未完成，完整模型逐层执行仍未派发，不以输入preflight替代该依赖验收。

## 原精度诊断的第三条天然对照

原精确容器仍运行中，保存logits审计已覆盖15/30条件，见
`omni-precision-train6-v1-logit-audit-partial-08.json`。第三条（row2）在既有输入
预检查中的左padding数为0；其reference→autocast→FP32 features→audio padding→
text padding四个相邻比较，完整BF16 logits逐元素相同，FP32响应logp最大差均为0。
这一无新增左padding对照与前两条text padding条件出现差异的观察一致，但尚不能
区分具体SDPA/MoE/算子形状数值机制，也不改变原0.1概率一致性门槛。
本次仅审计已保存tensor，没有重跑模型；全六条诊断和后续逐层定位仍未完成。
