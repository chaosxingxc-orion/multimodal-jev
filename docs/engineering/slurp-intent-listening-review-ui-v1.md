# 独立开发候选的听音审核材料

2026-11-12 UTC。固定300条selection来源的参考文本筛查与文件信号检查已完成，但缺少音频语义对齐及独立本体复审。本轮准备自包含离线HTML，可播放每条原音频、显示参考文本和原意图、记录审核人/对齐状态/标签判断/理由，并导出JSON；未自动授予评测或训练资格。

页面路径：`/mnt/cephfs/xingchao/datasets/slurp-jev-benchmark/intent-development-review-v1/listening-review-v1.html`。约37.3MB，300条音频均内嵌，无网络上传、无模型预测或推荐答案。参考筛查中的211/89状态未显示给听音人，避免把先前筛查结论作为音频裁决。全部300条保留；导出中的未审核项为pending，所有evaluation_allowed和optimization_allowed均为false。

实际校验全部300段内嵌WAV与原FLAC解码后的PCM16样本逐一相等。JS通过node语法检查，另有参考文本脚本注入/JSON往返测试；没有安装浏览器自动化，尚未执行实际浏览器交互验证。材料构建不是听音审核，未将任何样本提升为可评测。原始音频/记录/packet哈希保留在审核记录中，防止审核结论误绑定；后续应独立校验导入记录、复审本体边界，再发布新的固定评测版本。

证据见同名JSON与`slurp-intent-listening-review-ui-v1-verified.json`。该材料不改变89条学习发布，不读取官方TEST，也不改变训练奖励。

补充导入工具`tools/jev/verify_intent_listening_review.py`核对全部原始样本、packet/参考审核/音频/记录哈希，以及审核人、带时区时间、明确状态和理由。8项测试覆盖错绑、缺少依据/审核人、非法时间和擅自准入。即使填写完整，输出仍强制`evaluation_allowed=false`、`optimization_allowed=false`；这是账本完整性校验，不能替代独立语义裁决。当前没有导入实际听音审核结果。
