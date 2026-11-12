# 开发候选集参考文本首轮筛查

2026-11-12 UTC。逐条读取固定300条开发候选的官方参考文本与canonical intent，211条判为reference_consistent、89条pending。参考相符子集覆盖47类，原候选集覆盖52类。所有300条都保留在逐条账本中，不覆盖原标签、不补替代gold、不按模型输出筛选。

账本为`configs/reviews/slurp-intent-development-reference-audit-v1.json`，每条绑定录音ID、source_group、原记录内容哈希、意图和理由。独立检查已验证完整覆盖、原始身份和计数；同名JSON是账本完整性证明，不是语义正确性证明。

待定主要包括：交通票价/票务搜索与购票混淆；泛化的`next one`缺少播客证据；调灯亮度与颜色变更混淆；未说明对象的控制请求；普通事实/新闻/建议与`general_quirky`边界。明确文本与标签冲突时仅隔离，不将推断的另一个标签作为新gold。

此次为模型代理的离线参考文本筛查，未复听音频或进行音频/文本对齐裁决，也未完成独立本体专家复审。reference_consistent不等于已审核基准。账本及校验器强制`audio_reviewed=false`、`evaluation_allowed=false`、`optimization_allowed=false`，因此当前不能用于训练、奖励、模型选择或正式收益报告。

后续应核对音频/参考对齐、复审边界并发布固定开发协议；记录筛选前后的类覆盖及分母，避免将排除89条造成的变化称为算法提升。该集来自官方TRAIN的内部selection来源组，不声称官方TEST或说话人隔离。
