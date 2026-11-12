# FSC官方TEST完整输入处理检查

2026-11-12 UTC。在固定镜像`sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c`启动CPU容器，限定2 CPU、8GB内存、无网络、无NPU设备。实际容器`4b212fad07b7bf1e17445ff041b034d3391a7df792a599692cb7ef6dda46055c`在启动核验时为running；收据见`fsc-official-test-processor-cpu-v1-launch.json`。本记录不表示已完成。

输入为已发布且哈希固定的3793条官方TEST音频身份名单（10位说话人）。逐条生成direct/structured × native/permutation/rebind，共22758请求；检查完整模板不超过4096tokens、候选符号可编码、全部视图具有相同音频特征身份，并导出后续HF/vLLM可直接消费的输入记录。复用原12条TRAIN处理逻辑，新增明确的官方名单角色，拒绝重复身份、条数/说话人数不符以及混入gold等字段的名单。原工程12条契约保留。35项处理名单、官方人口、后端身份相关测试通过。

任务配置、名单、processor文件及源码均固定哈希。程序只读取发布后的音频名单，不调用官方CSV标签提取器，不加载模型权重，不输出预测或目标域分数。官方划分完整性由既有名单发布负责，本工具不把自报的recordings数作为新的官方完整性认证。

共享根目录为`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-official-test-processor-cpu-v1`，结果预期为`run/result.json`，成功需实际Docker退出0和`run/terminal.json`相符。失败/超时保留原容器与终态，不自动替换或删样本规避。该检查独立于D/Q/P NPU队列，不选择源域checkpoint，也不授予迁移收益声明。

完成更新：实际Docker退出0，全部3793条、10位说话人、22758请求已处理。最长完整输入3031tokens，六视图各3793条，无截断或缺失。独立复核逐条模型输入与固定task编译结果、绑定、音频哈希、同录音跨视图音频特征/处理身份及请求覆盖，全部通过；文件pins与终态一致。见`fsc-official-test-processor-cpu-v1-verified.json`。没有模型权重推理或目标标签评分，该产物现可作为后续正式目标gate的固定CPU输入。
