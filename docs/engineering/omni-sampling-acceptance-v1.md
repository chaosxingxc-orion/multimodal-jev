# 冻结 Omni 采样独立验收

`src/multimodal_jev/omni_sampling_acceptance.py` 直接读取原始采样工件，
独立重建转换后的记录，不调用采样侧 `capture`。`tools/jev/verify_sfc_omni_sampling.py`
在此基础上核查精确 Docker 终态、镜像、源码、模型身份与 processor 资产。

## 覆盖范围

- 必须为原始四个学习会话、六个回合、78 个决策，每个决策两个 replicate。
- 每个决策必须恰有 intent、raw、result 三个文件，共 234 个；缺失或多余均拒绝。
- 原始 input token 必须与 processor 预检哈希相同，且预留 1024 输出 token 后不超 4096。
- 每个原始响应必须有唯一的编号 0 或 1、完整 token、有限非正采样 logprob、
  stop/length 终态。完成顺序可以颠倒，不得改变 replicate 身份。
- 原始文本须等于 tokenizer 对完整响应 token 的解码；拒绝文本修复或概率补写。
- 独立重建的 156 条记录必须与逐请求转换结果、最终 records.jsonl 完全相等。
- 检查 summary/identity/terminal 哈希绑定、运行成本、完整数量及零权重更新声明。
- 容器必须精确匹配 launch 的完整 ID、固定镜像、退出 0 且非 OOM。
  实际映射的计算卡必须恰为 NPU6/7，拒绝 privileged 容器；仅检查计划声明不够。
- preparation 必须列出快照全部 src/multimodal_jev 与 tools/jev Python 文件，
  验收代码和运行快照逐文件相同。配置、模型资产清单、版本、提示和采样参数全部绑定。

验收器不读目标标签、不计算任务分数、不执行推理，也不重试未知状态请求。
tokenizer 须在固定 Transformers 环境下加载；运行器已有 CPU 预检不能替代真实采样验收。

## 验证

82 项相关测试通过，其中新增独立验收测试包括完整 78×2 合成工件、20 类工件损坏、
精确容器/设备的正例及 8 类负例。合成数据的“model_loaded=true”只是测试字段，
没有实际模型运行。容器检查使用模拟 inspect 输出；真实 Docker 终态仍需派发后验证。

本轮还修复 `test_sfc_omni_sampling.py` 的测试顺序依赖：它此前依赖其他测试先将
tools/jev 放进导入路径，现在使用 scoped monkeypatch 独立配置路径。

## 待接入项

当前尚无真实 156 条采样结果，独立验收不能声称已通过真实推理。
调度准备须提供 preparation.json、launch.json、不可变 source 和完整 source_files；
队列还需等待现有正式基线结束、占用一致的资源锁，并提供精确容器监督。
跨环境验收还需要固定 tokenizer 环境及只读共享文件访问。

资产前后逐文件哈希校验由已固定的采样运行器执行；验收器核查资产清单与源码身份，
不重复读取全部权重。检查零更新字段与固定源码也不等同证明任意外部程序未改模型。
HF/vLLM 实际逐 token 概率误差、thinker 前向内存及训练能力仍需独立验证。
