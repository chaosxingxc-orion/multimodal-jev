# FSC真实音频配置迁移输入验证

2026-11-12 UTC。此前仅有合成静音processor检查，本轮固定从官方TRAIN的23,132个不同路径中按seed941哈希选12条录音，不使用类别、转写、模型输出筛选。CSV解析仅提取path，完整CSV身份哈希保留；未提取目标标签进行评分，未读取valid/test。

每条音频编译direct/structured × native/permutation/rebind，共72个请求。使用已冻结`fsc-span-transfer-v1`配置，保持31联合命令、6/14/4组件候选，seed317绑定。真实音频直接进入Qwen3-ASR processor，候选定义/要求均完整输入；超过4096会失败，不静默截断。六种配置的音频前处理回执、input_features和mask一致。

固定训练镜像的独立CPU容器运行，未挂载NPU、非privileged，实际退出0。独立重新编译全部72条模型输入和语义绑定，核对音频哈希、候选支持及全量名单成功；最大输入长度见`fsc-real-span-processor-cpu-v1-verified.json`。模型权重没有加载，实际NPU完整候选概率仍未验收；这是配置可消费性准备，不是目标域成绩或迁移收益。

该input-only artifact保留完整输入身份，可供之后模型冻结后的后端验证使用；当前SLURP连续RL优先，不占用训练卡派发FSC模型推理。目标模板仍不依赖FSC标签反馈，后续正式迁移要固定源域checkpoint选择规则和目标评测划分，不能拿此12条工程TRAIN样本当测试集。
