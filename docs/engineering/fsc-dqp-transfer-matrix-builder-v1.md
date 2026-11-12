# 完整FSC迁移矩阵准备入口

2026-11-12 UTC。`prepare_fsc_transfer_matrix.py`在源域D/Q/P全部完成后，重新调用真实源域绑定器，确认选定七策略的实际Docker退出、完整178输入验收、adapter内容和arm/update。随后验证全量CPU输入的实际退出、任务/名单身份、六视图覆盖、音频文件及处理特征一致性。任何源域未完成时不创建目标矩阵目录。

通过后将任务与3793条音频人口复制到共享评测根，生成实际source-selection和评分入口可消费的protocol，再逐一生成7个独立gate。各gate配置固定protocol、task、population、source-selection、CPU输入、模型及adapter和自身源码，按实际Docker挂载验证全部pins可见。模型沿用Qwen3-ASR-0.6B，BF16未合并adapter，NPU4 TP1、4096上下文及已验收short-audio-v2 profile，不引入文本解码或目标标签。

每策略22758请求、HF和vLLM各执行一次，七策略合计159306请求（两后端计算另计）。小样本模板3600秒上限扩为每策略172800秒固定上限；该上限不是完成时长预测，也不授权失败自动重跑。实际执行仍须按资源队列安排，当前没有启动任何目标模型推理。

25项矩阵准备/源域绑定/评分入口测试通过。完整准备测试使用模拟已验收源域和CPU前提，实际生成7套源码/配置并检查protocol与挂载pins、完整人口、相同adapter内容允许保留、目录不覆盖及无容器启动。该模拟不能替代未来真实检查点的验收。

准备器配置位于`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-dqp-transfer-matrix-builder-v1/config.json`，SHA256 `d3fc061b816a1f81759ce2d6cd4ab8960570a05561eca6c4c8688fe1bc2f4b7f`。未来输出根`/mnt/cephfs/xingchao/experiments/multimodal-jev/fsc-dqp-transfer-matrix-v1`当前不存在。source-selection必须在真实源域完成后产生；不预填未来adapter哈希，也不从FSC结果挑选检查点。
