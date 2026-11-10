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
