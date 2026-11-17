# 首批原生意图监督与 D/Q/P 请求

2026-11-12 UTC。对预先按类均衡抽出的120个独立学习来源逐条审核官方参考文本与canonical scenario/action的语义对应。89个意图判决准入，31个保留pending；89个来源覆盖56类。四个未准入类为general_quirky、music_settings、play_podcasts、qa_maths。全部60类仍留在模型候选中，不通过缩小候选空间提高指标。

审核依据及每条理由见 `configs/reviews/slurp-intent-learning-reference-audit-v1.json`。这是模型代理进行的参考文本语义审核，未复听音频或核查音频/参考对齐，也不是完整官方本体边界认证。参考文本仅用于离线审核，未进入actor输入；槽位类型未准入。模糊对象、跨意图边界和上下文依赖样本不强行补标签。学习集之外的标签未打开。

发布使用既有 `slurp_training_release.publish`：绑定源清单、任务、原始target、审核及音频哈希；复核89条的逐判决准入、89个独立来源及31个pending排除。发布位置与哈希见同名JSON。

`slurp_intent_structure.py` 从同一任务配置编译 D/Q/P：D仅意图Final，Q/P为scenario、action、同一个意图Final。Q与P输入逐字节相同；监督权重分别为D `[1]`、Q `[0,0,1]`、P `[0.1,0.1,1]`。完整问题及候选在输入前固定，target保持独立。过程标签仅由已审核joint intent的canonical pair投影，不能称为独立过程标注或已验证多跳推理。

配置产物见 `slurp-intent-dqp-pilot-v1.json`：各臂89请求，全部候选不变。原始样本每轮一次，仅使用外层逆类频率权重（有效权重总和89），不叠加均衡重采样。该产物尚未完成processor上下文和真实后端检查，不能直接宣称可运行训练或有效泛化。正式学习曲线还需扩大来源与准备独立审核评测。
